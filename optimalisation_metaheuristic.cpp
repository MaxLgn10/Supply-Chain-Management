/*
optimalisation_metaheuristic.cpp

C++ translation of optimalisation_metaheuristic.py.

Purpose:
  Metaheuristic pre-pack optimizer for the SCM group assignment. It jointly
  optimizes pack contents and pack allocations using demand-curve construction,
  randomized greedy repair, ALNS-style neighborhood operators, and simulated
  annealing acceptance.

Important translation notes:
  - This C++ version is intentionally dependency-light and uses only the C++17
    standard library.
  - The Python version used pandas and openpyxl for Excel input/output. Standard
    C++ has no built-in Excel reader/writer, so this translation supports CSV
    input and CSV output. If --products points to a CSV with columns id,cost,
    product costs are loaded; otherwise all unit costs are set to 1.
  - The official XLSX solution template filling from the Python version is not
    implemented here. The optimizer writes the same main CSV outputs.

Compile:
  g++ -std=c++17 -O2 -Wall -Wextra -pedantic optimalisation_metaheuristic.cpp -o
optimalisation_metaheuristic

Run:
  ./optimalisation_metaheuristic

Recommended stronger run:
  ./optimalisation_metaheuristic --iterations 50000 --restarts 8 --time-limit
1800
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// =============================================================================
// Cost parameters from the current assignment code base
// =============================================================================

static constexpr double COST_OF_CAPITAL = 0.243;
static constexpr double HANDLING_COST_PER_PACK = 11.03;
static constexpr double PACK_CREATION_COST = 134.00;
static constexpr double SHORTAGE_PENALTY_PER_UNIT = 10000.00;

static constexpr int MAX_PACK_UNITS = 10000;
static constexpr int MAX_DISTINCT_SKUS_PER_PACK = 10000;
static constexpr int MAX_REPAIR_STEPS_PER_CHANNEL = 20000;
static constexpr int PRUNE_PASSES = 3;

// =============================================================================
// Data structures
// =============================================================================

struct SkuMeta {
  std::string sku_id;
  std::string product_id;
  std::string size;
  std::string category_group;
  bool has_category = false;
};

struct ProblemData {
  std::vector<std::vector<int>> demand; // demand[sku][channel]
  std::vector<std::string> sku_ids;
  std::vector<std::string> channel_ids;
  std::vector<double> unit_cost;
  std::vector<SkuMeta> sku_meta;
  std::unordered_map<std::string, int> sku_index;
  std::unordered_map<std::string, int> channel_index;
};

struct Solution {
  std::vector<std::vector<int>> content; // content[pack][sku]
  std::vector<std::vector<int>> alloc;   // alloc[pack][channel]
  std::vector<std::string> names;

  double cost = 0.0;
  double setup_cost = 0.0;
  double handling_cost = 0.0;
  double capital_cost = 0.0;
  double shortage_cost = 0.0;
  long long overstock_units = 0;
  long long shortage_units = 0;
  long long shipped_units = 0;
  long long allocated_packs = 0;
};

struct LogRow {
  int restart = 0;
  int iteration = 0;
  std::string op;
  double current_cost = 0.0;
  double restart_best_cost = 0.0;
  double global_best_cost = 0.0;
  double temperature = 0.0;
  int pack_types_current = 0;
  int pack_types_best = 0;
  long long shortage_units_best = 0;
  long long overstock_units_best = 0;
  double elapsed_seconds = 0.0;
};

struct Args {
  std::string forecast = "outputs/master_forecast_2026.csv";
  std::string products = "data/PPP_stu_products.csv";
  std::string template_path = "data/PPP_solutionFile_2026.xlsx";
  std::string output_dir = "outputs";
  int iterations = 20000;
  int restarts = 5;
  int seed = 42;
  double initial_temperature = 5000.0;
  double cooling_rate = 0.9995;
  int seed_packs = 250;
  std::optional<int> time_limit;
  std::optional<int> max_pack_types;
  int min_forecast = 1;
  bool export_solution = true;
};

// =============================================================================
// Utility functions
// =============================================================================

static std::string trim(const std::string &s) {
  std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return "";
  std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

static std::string make_sku_id(const std::string &product_id,
                               const std::string &size) {
  return product_id + "_" + size;
}

static std::string slugify(const std::string &value) {
  std::string lower;
  lower.reserve(value.size());
  for (unsigned char ch : value)
    lower.push_back(static_cast<char>(std::tolower(ch)));
  std::string out = std::regex_replace(lower, std::regex("[^a-z0-9]+"), "_");
  while (!out.empty() && out.front() == '_')
    out.erase(out.begin());
  while (!out.empty() && out.back() == '_')
    out.pop_back();
  return out.empty() ? "value" : out;
}

static int ceil_nonnegative(const std::string &value) {
  std::string t = trim(value);
  if (t.empty())
    return 0;
  try {
    double x = std::stod(t);
    if (std::isnan(x))
      return 0;
    return std::max(0, static_cast<int>(std::ceil(x)));
  } catch (...) {
    return 0;
  }
}

static std::vector<int> safe_int_vector(const std::vector<int> &v) {
  std::vector<int> out = v;
  for (int &x : out)
    x = std::max(0, x);
  return out;
}

static long long sum_vector(const std::vector<int> &v) {
  return std::accumulate(v.begin(), v.end(), 0LL);
}

static int count_nonzero(const std::vector<int> &v) {
  return static_cast<int>(
      std::count_if(v.begin(), v.end(), [](int x) { return x > 0; }));
}

static std::string csv_escape(const std::string &value) {
  bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
  if (!needs_quotes)
    return value;
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"')
      out += "\"\"";
    else
      out += ch;
  }
  out += "\"";
  return out;
}

static std::vector<std::string> parse_csv_line(const std::string &line) {
  std::vector<std::string> fields;
  std::string cur;
  bool in_quotes = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    char ch = line[i];
    if (in_quotes) {
      if (ch == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        cur.push_back(ch);
      }
    } else {
      if (ch == '"')
        in_quotes = true;
      else if (ch == ',') {
        fields.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(ch);
      }
    }
  }
  fields.push_back(cur);
  return fields;
}

static std::vector<std::unordered_map<std::string, std::string>>
read_csv_records(const fs::path &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot open CSV file: " + path.string());

  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("CSV file is empty: " + path.string());
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  std::vector<std::string> headers = parse_csv_line(line);
  for (auto &h : headers)
    h = trim(h);

  std::vector<std::unordered_map<std::string, std::string>> records;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    std::vector<std::string> fields = parse_csv_line(line);
    std::unordered_map<std::string, std::string> rec;
    for (std::size_t i = 0; i < headers.size(); ++i) {
      rec[headers[i]] = (i < fields.size()) ? fields[i] : "";
    }
    records.push_back(std::move(rec));
  }
  return records;
}

static std::vector<std::string>
ensure_unique_names(const std::vector<std::string> &names) {
  std::unordered_map<std::string, int> seen;
  std::vector<std::string> out;
  out.reserve(names.size());
  for (const auto &raw : names) {
    std::string base = raw.empty() ? "pack" : raw;
    int n = ++seen[base];
    if (n == 1)
      out.push_back(base);
    else
      out.push_back(base + "_" + std::to_string(n));
  }
  return out;
}

static double
seconds_since(const std::chrono::steady_clock::time_point &start) {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now() - start).count();
}

static int random_index_weighted(std::mt19937 &rng,
                                 const std::vector<double> &weights) {
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return dist(rng);
}

static int randint(std::mt19937 &rng, int lo, int hi_inclusive) {
  std::uniform_int_distribution<int> dist(lo, hi_inclusive);
  return dist(rng);
}

static double rand01(std::mt19937 &rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

static std::vector<int>
proportional_integer_vector(const std::vector<double> &weights, int total_units,
                            int max_distinct, std::mt19937 &rng) {
  int n = static_cast<int>(weights.size());
  std::vector<int> result(n, 0);
  if (total_units <= 0)
    return result;

  std::vector<int> positive_idx;
  for (int i = 0; i < n; ++i) {
    if (weights[i] > 0.0)
      positive_idx.push_back(i);
  }
  if (positive_idx.empty())
    return result;

  std::vector<std::pair<int, double>> scored;
  scored.reserve(positive_idx.size());
  std::uniform_real_distribution<double> jitter(0.90, 1.10);
  for (int idx : positive_idx)
    scored.push_back({idx, weights[idx] * jitter(rng)});
  std::sort(scored.begin(), scored.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  int keep_count =
      std::min({max_distinct, total_units, static_cast<int>(scored.size())});
  std::vector<int> kept;
  kept.reserve(keep_count);
  for (int i = 0; i < keep_count; ++i)
    kept.push_back(scored[i].first);

  double total_weight = 0.0;
  for (int idx : kept)
    total_weight += weights[idx];
  if (total_weight <= 0.0)
    return result;

  std::vector<double> raw(keep_count, 0.0);
  std::vector<int> base(keep_count, 0);
  for (int i = 0; i < keep_count; ++i) {
    raw[i] = weights[kept[i]] / total_weight * total_units;
    base[i] = static_cast<int>(std::floor(raw[i]));
  }

  if (total_units >= keep_count) {
    for (int &x : base)
      x = std::max(1, x);
  }

  int diff = total_units - static_cast<int>(sum_vector(base));
  if (diff > 0) {
    std::vector<int> order(keep_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      double ra = raw[a] - std::floor(raw[a]);
      double rb = raw[b] - std::floor(raw[b]);
      return ra > rb;
    });
    for (int k = 0; k < diff && k < keep_count; ++k)
      base[order[k]]++;
  } else if (diff < 0) {
    std::vector<int> order(keep_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return base[a] > base[b]; });
    for (int k : order) {
      while (diff < 0 && base[k] > 1) {
        base[k]--;
        diff++;
      }
      if (diff == 0)
        break;
    }
  }

  for (int i = 0; i < keep_count; ++i) {
    if (base[i] > 0)
      result[kept[i]] = base[i];
  }
  return result;
}

// =============================================================================
// Loading data
// =============================================================================

static std::unordered_map<std::string, double>
load_product_costs_if_csv(const fs::path &product_path) {

  std::unordered_map<std::string, double> costs;
  if (!fs::exists(product_path))
    return costs;

  std::string ext = product_path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (ext != ".csv")
    return costs;

  auto records = read_csv_records(product_path);
  for (const auto &rec : records) {
    auto id_it = rec.find("id");
    auto cost_it = rec.find("cost");
    if (id_it == rec.end() || cost_it == rec.end())
      continue;
    try {
      costs[id_it->second] = std::stod(cost_it->second);
    } catch (...) {
      // Ignore malformed cost.
    }
  }
  return costs;
}

static ProblemData load_problem(const std::string &forecast_path,
                                const std::string &product_path,
                                int min_forecast) {
  fs::path forecast_file(forecast_path);
  if (!fs::exists(forecast_file)) {
    throw std::runtime_error("Forecast file not found: " + forecast_path +
                             ". Run run_forecasting.py first to create "
                             "outputs/master_forecast_2026.csv.");
  }

  auto records = read_csv_records(forecast_file);
  const std::vector<std::string> required = {"channel_id", "product_id", "size",
                                             "forecast_ensemble"};
  for (const auto &col : required) {
    if (records.empty() || records.front().find(col) == records.front().end()) {
      throw std::runtime_error("Forecast file is missing required column: " +
                               col);
    }
  }

  bool has_category =
      !records.empty() &&
      records.front().find("category_group") != records.front().end();

  std::map<std::pair<std::string, std::string>, int> grouped_demand;
  std::unordered_map<std::string, SkuMeta> meta_by_sku;
  std::set<std::string> channels_set;

  for (const auto &rec : records) {
    std::string channel = rec.at("channel_id");
    std::string product_id = rec.at("product_id");
    std::string size = rec.at("size");
    std::string sku = make_sku_id(product_id, size);
    int forecast_units = ceil_nonnegative(rec.at("forecast_ensemble"));

    grouped_demand[{sku, channel}] += forecast_units;
    channels_set.insert(channel);

    if (meta_by_sku.find(sku) == meta_by_sku.end()) {
      SkuMeta meta;
      meta.sku_id = sku;
      meta.product_id = product_id;
      meta.size = size;
      meta.has_category = has_category;
      if (has_category) {
        auto it = rec.find("category_group");
        meta.category_group = (it == rec.end()) ? "" : it->second;
      }
      meta_by_sku[sku] = meta;
    }
  }

  std::map<std::string, int> total_by_sku;
  for (const auto &kv : grouped_demand)
    total_by_sku[kv.first.first] += kv.second;

  ProblemData data;
  for (const auto &kv : total_by_sku) {
    if (kv.second >= min_forecast)
      data.sku_ids.push_back(kv.first);
  }
  if (data.sku_ids.empty())
    throw std::runtime_error(
        "No SKU has positive forecast demand after filtering.");

  data.channel_ids.assign(channels_set.begin(), channels_set.end());
  for (int i = 0; i < static_cast<int>(data.sku_ids.size()); ++i)
    data.sku_index[data.sku_ids[i]] = i;
  for (int i = 0; i < static_cast<int>(data.channel_ids.size()); ++i)
    data.channel_index[data.channel_ids[i]] = i;

  int n_skus = static_cast<int>(data.sku_ids.size());
  int n_channels = static_cast<int>(data.channel_ids.size());
  data.demand.assign(n_skus, std::vector<int>(n_channels, 0));
  data.sku_meta.reserve(n_skus);

  for (const std::string &sku : data.sku_ids)
    data.sku_meta.push_back(meta_by_sku.at(sku));

  for (const auto &kv : grouped_demand) {
    const std::string &sku = kv.first.first;
    const std::string &channel = kv.first.second;
    auto si = data.sku_index.find(sku);
    auto ci = data.channel_index.find(channel);
    if (si != data.sku_index.end() && ci != data.channel_index.end()) {
      data.demand[si->second][ci->second] = kv.second;
    }
  }

  auto cost_by_product = load_product_costs_if_csv(product_path);
  data.unit_cost.assign(n_skus, 1.0);
  if (!cost_by_product.empty()) {
    for (int i = 0; i < n_skus; ++i) {
      auto it = cost_by_product.find(data.sku_meta[i].product_id);
      if (it != cost_by_product.end())
        data.unit_cost[i] = it->second;
    }
  } else {
    std::cerr << "WARNING: product costs not loaded. Supply a CSV with id,cost "
                 "columns to use costs; using unit cost = 1.\n";
  }

  return data;
}

// =============================================================================
// Cost evaluation
// =============================================================================

static std::vector<std::vector<int>> shipped_matrix(const Solution &sol,
                                                    const ProblemData &data) {
  int n_skus = static_cast<int>(data.sku_ids.size());
  int n_channels = static_cast<int>(data.channel_ids.size());
  std::vector<std::vector<int>> shipped(n_skus,
                                        std::vector<int>(n_channels, 0));

  for (std::size_t p = 0; p < sol.content.size(); ++p) {
    for (int s = 0; s < n_skus; ++s) {
      int q = sol.content[p][s];
      if (q == 0)
        continue;
      for (int c = 0; c < n_channels; ++c)
        shipped[s][c] += q * sol.alloc[p][c];
    }
  }
  return shipped;
}

static double evaluate(Solution &sol, const ProblemData &data) {
  int n_skus = static_cast<int>(data.sku_ids.size());
  int n_channels = static_cast<int>(data.channel_ids.size());
  auto shipped = shipped_matrix(sol, data);

  long long allocated_packs = 0;
  int active_pack_types = 0;
  for (std::size_t p = 0; p < sol.content.size(); ++p) {
    long long content_sum = sum_vector(sol.content[p]);
    long long alloc_sum = sum_vector(sol.alloc[p]);
    allocated_packs += alloc_sum;
    if (content_sum > 0 && alloc_sum > 0)
      active_pack_types++;
  }

  long long over_units = 0;
  long long under_units = 0;
  long long shipped_units = 0;
  double capital_cost = 0.0;
  for (int s = 0; s < n_skus; ++s) {
    for (int c = 0; c < n_channels; ++c) {
      int sent = shipped[s][c];
      int dem = data.demand[s][c];
      shipped_units += sent;
      int over = std::max(0, sent - dem);
      int under = std::max(0, dem - sent);
      over_units += over;
      under_units += under;
      capital_cost += over * data.unit_cost[s] * COST_OF_CAPITAL;
    }
  }

  sol.setup_cost = active_pack_types * PACK_CREATION_COST;
  sol.handling_cost = allocated_packs * HANDLING_COST_PER_PACK;
  sol.capital_cost = capital_cost;
  sol.shortage_cost = under_units * SHORTAGE_PENALTY_PER_UNIT;
  sol.cost =
      sol.setup_cost + sol.handling_cost + sol.capital_cost + sol.shortage_cost;
  sol.overstock_units = over_units;
  sol.shortage_units = under_units;
  sol.shipped_units = shipped_units;
  sol.allocated_packs = allocated_packs;
  return sol.cost;
}

static double delta_add_one_pack(const std::vector<int> &content_vec,
                                 const std::vector<int> &shipped_col,
                                 const std::vector<int> &demand_col,
                                 const std::vector<double> &unit_cost) {
  double delta_capital = 0.0;
  long long delta_shortage_units = 0;
  for (std::size_t s = 0; s < content_vec.size(); ++s) {
    int before_over = std::max(0, shipped_col[s] - demand_col[s]);
    int before_under = std::max(0, demand_col[s] - shipped_col[s]);
    int after = shipped_col[s] + content_vec[s];
    int after_over = std::max(0, after - demand_col[s]);
    int after_under = std::max(0, demand_col[s] - after);
    delta_capital +=
        (after_over - before_over) * unit_cost[s] * COST_OF_CAPITAL;
    delta_shortage_units += (after_under - before_under);
  }
  return HANDLING_COST_PER_PACK + delta_capital +
         delta_shortage_units * SHORTAGE_PENALTY_PER_UNIT;
}

static double delta_remove_one_pack(const std::vector<int> &content_vec,
                                    const std::vector<int> &shipped_col,
                                    const std::vector<int> &demand_col,
                                    const std::vector<double> &unit_cost) {
  double delta_capital = 0.0;
  long long delta_shortage_units = 0;
  for (std::size_t s = 0; s < content_vec.size(); ++s) {
    int before_over = std::max(0, shipped_col[s] - demand_col[s]);
    int before_under = std::max(0, demand_col[s] - shipped_col[s]);
    int after = shipped_col[s] - content_vec[s];
    int after_over = std::max(0, after - demand_col[s]);
    int after_under = std::max(0, demand_col[s] - after);
    delta_capital +=
        (after_over - before_over) * unit_cost[s] * COST_OF_CAPITAL;
    delta_shortage_units += (after_under - before_under);
  }
  return -HANDLING_COST_PER_PACK + delta_capital +
         delta_shortage_units * SHORTAGE_PENALTY_PER_UNIT;
}

// =============================================================================
// Construction and repair
// =============================================================================

static Solution remove_empty_pack_rows(const Solution &sol) {
  if (sol.content.empty())
    return sol;
  Solution out;
  for (std::size_t p = 0; p < sol.content.size(); ++p) {
    bool used_content = sum_vector(sol.content[p]) > 0;
    bool used_alloc = sum_vector(sol.alloc[p]) > 0;
    if (used_content || used_alloc) {
      out.content.push_back(sol.content[p]);
      out.alloc.push_back(sol.alloc[p]);
      out.names.push_back(sol.names[p]);
    }
  }
  return out;
}

static Solution create_single_sku_initial_solution(ProblemData &data) {
  int n_skus = static_cast<int>(data.sku_ids.size());
  int n_channels = static_cast<int>(data.channel_ids.size());
  Solution sol;
  sol.content.assign(n_skus, std::vector<int>(n_skus, 0));
  sol.alloc.assign(n_skus, std::vector<int>(n_channels, 0));
  sol.names.reserve(n_skus);
  for (int s = 0; s < n_skus; ++s) {
    sol.content[s][s] = 1;
    sol.alloc[s] = data.demand[s];
    sol.names.push_back("single_" + slugify(data.sku_ids[s]));
  }
  evaluate(sol, data);
  return sol;
}

static std::vector<int> make_curve_pack_from_indices(
    const std::vector<int> &indices, const std::vector<double> &weights,
    std::mt19937 &rng, int min_units = 4, int max_units = MAX_PACK_UNITS) {
  std::vector<int> zero(weights.size(), 0);
  if (indices.empty())
    return zero;

  std::vector<int> choices = {4, 6, 8, 10, 12, 16, 20, 24};
  int total = choices[randint(rng, 0, static_cast<int>(choices.size()) - 1)];
  total = std::max(min_units, std::min(max_units, total));

  std::vector<double> local_weights(weights.size(), 0.0);
  for (int idx : indices)
    local_weights[idx] = weights[idx];
  return proportional_integer_vector(
      local_weights, total,
      std::min(MAX_DISTINCT_SKUS_PER_PACK, static_cast<int>(indices.size())),
      rng);
}

static std::pair<std::vector<int>, std::string>
create_random_curve_pack(const ProblemData &data, std::mt19937 &rng) {
  int n_skus = static_cast<int>(data.sku_ids.size());
  int n_channels = static_cast<int>(data.channel_ids.size());
  std::vector<double> demand_total(n_skus, 0.0);
  for (int s = 0; s < n_skus; ++s)
    demand_total[s] = static_cast<double>(sum_vector(data.demand[s]));

  std::vector<std::string> modes = {"product", "channel_product", "category",
                                    "channel", "global"};
  std::string mode = modes[randint(rng, 0, static_cast<int>(modes.size()) - 1)];

  if (mode == "product") {
    std::vector<std::string> products;
    for (const auto &m : data.sku_meta)
      products.push_back(m.product_id);
    std::sort(products.begin(), products.end());
    products.erase(std::unique(products.begin(), products.end()),
                   products.end());
    if (!products.empty()) {
      std::string pid =
          products[randint(rng, 0, static_cast<int>(products.size()) - 1)];
      std::vector<int> idx;
      for (int i = 0; i < n_skus; ++i)
        if (data.sku_meta[i].product_id == pid)
          idx.push_back(i);
      return {make_curve_pack_from_indices(idx, demand_total, rng),
              "prod_" + pid + "_curve"};
    }
  }

  if (mode == "channel_product") {
    int ch_idx = randint(rng, 0, n_channels - 1);
    std::vector<std::string> positive_products;
    for (int s = 0; s < n_skus; ++s) {
      if (data.demand[s][ch_idx] > 0)
        positive_products.push_back(data.sku_meta[s].product_id);
    }
    std::sort(positive_products.begin(), positive_products.end());
    positive_products.erase(
        std::unique(positive_products.begin(), positive_products.end()),
        positive_products.end());
    if (!positive_products.empty()) {
      std::string pid = positive_products[randint(
          rng, 0, static_cast<int>(positive_products.size()) - 1)];
      std::vector<int> idx;
      std::vector<double> weights(n_skus, 0.0);
      for (int s = 0; s < n_skus; ++s) {
        weights[s] = static_cast<double>(data.demand[s][ch_idx]);
        if (data.sku_meta[s].product_id == pid && data.demand[s][ch_idx] > 0)
          idx.push_back(s);
      }
      return {make_curve_pack_from_indices(idx, weights, rng),
              "prod_" + pid + "_" + slugify(data.channel_ids[ch_idx]) +
                  "_curve"};
    }
  }

  if (mode == "category") {
    std::vector<std::string> categories;
    for (const auto &m : data.sku_meta)
      if (m.has_category && !m.category_group.empty())
        categories.push_back(m.category_group);
    std::sort(categories.begin(), categories.end());
    categories.erase(std::unique(categories.begin(), categories.end()),
                     categories.end());
    if (!categories.empty()) {
      std::string cat =
          categories[randint(rng, 0, static_cast<int>(categories.size()) - 1)];
      std::vector<int> idx;
      for (int s = 0; s < n_skus; ++s)
        if (data.sku_meta[s].category_group == cat)
          idx.push_back(s);
      return {make_curve_pack_from_indices(idx, demand_total, rng),
              "cat_" + slugify(cat) + "_curve"};
    }
  }

  if (mode == "channel") {
    int ch_idx = randint(rng, 0, n_channels - 1);
    std::vector<int> idx;
    std::vector<double> weights(n_skus, 0.0);
    for (int s = 0; s < n_skus; ++s) {
      weights[s] = static_cast<double>(data.demand[s][ch_idx]);
      if (data.demand[s][ch_idx] > 0)
        idx.push_back(s);
    }
    return {make_curve_pack_from_indices(idx, weights, rng),
            "channel_" + slugify(data.channel_ids[ch_idx]) + "_curve"};
  }

  std::vector<int> idx;
  for (int s = 0; s < n_skus; ++s)
    if (demand_total[s] > 0.0)
      idx.push_back(s);
  return {make_curve_pack_from_indices(idx, demand_total, rng), "global_curve"};
}

static Solution append_pack(Solution sol, std::vector<int> pack,
                            const std::string &name) {
  if (sum_vector(pack) <= 0)
    return sol;
  for (int &x : pack)
    x = std::max(0, x);

  while (sum_vector(pack) > MAX_PACK_UNITS) {
    auto it = std::max_element(pack.begin(), pack.end());
    if (it == pack.end() || *it <= 0)
      break;
    (*it)--;
  }

  if (count_nonzero(pack) > MAX_DISTINCT_SKUS_PER_PACK) {
    std::vector<int> order(pack.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return pack[a] > pack[b]; });
    std::vector<int> new_pack(pack.size(), 0);
    for (int i = 0; i < MAX_DISTINCT_SKUS_PER_PACK; ++i)
      new_pack[order[i]] = pack[order[i]];
    pack = std::move(new_pack);
  }

  for (const auto &existing : sol.content) {
    if (existing == pack)
      return sol;
  }

  int n_channels =
      sol.alloc.empty() ? 0 : static_cast<int>(sol.alloc.front().size());
  if (n_channels == 0)
    throw std::runtime_error("Solution has no channel dimension.");

  sol.content.push_back(std::move(pack));
  sol.alloc.push_back(std::vector<int>(n_channels, 0));
  sol.names.push_back(name);
  sol.names = ensure_unique_names(sol.names);
  return sol;
}

static Solution repair_allocation(Solution sol, const ProblemData &data,
                                  std::mt19937 &rng) {
  if (sol.content.empty())
    return sol;

  for (auto &row : sol.content)
    for (int &x : row)
      x = std::max(0, x);
  for (auto &row : sol.alloc)
    for (int &x : row)
      x = std::max(0, x);

  Solution filtered;
  for (std::size_t p = 0; p < sol.content.size(); ++p) {
    if (sum_vector(sol.content[p]) > 0) {
      filtered.content.push_back(sol.content[p]);
      filtered.alloc.push_back(sol.alloc[p]);
      filtered.names.push_back(sol.names[p]);
    }
  }
  sol = std::move(filtered);
  if (sol.content.empty())
    return sol;

  int n_channels = static_cast<int>(data.channel_ids.size());
  int n_skus = static_cast<int>(data.sku_ids.size());
  auto shipped = shipped_matrix(sol, data);

  for (int c_idx = 0; c_idx < n_channels; ++c_idx) {
    int steps = 0;
    while (steps < MAX_REPAIR_STEPS_PER_CHANNEL) {
      std::vector<int> shortage(n_skus, 0);
      int max_shortage = 0;
      for (int s = 0; s < n_skus; ++s) {
        shortage[s] = data.demand[s][c_idx] - shipped[s][c_idx];
        max_shortage = std::max(max_shortage, shortage[s]);
      }
      if (max_shortage <= 0)
        break;

      std::vector<std::tuple<double, double, int>> best;
      std::vector<int> shipped_col(n_skus), demand_col(n_skus);
      for (int s = 0; s < n_skus; ++s) {
        shipped_col[s] = shipped[s][c_idx];
        demand_col[s] = data.demand[s][c_idx];
      }

      for (int p_idx = 0; p_idx < static_cast<int>(sol.content.size());
           ++p_idx) {
        const auto &pack = sol.content[p_idx];
        if (sum_vector(pack) <= 0)
          continue;

        int covered_shortage = 0;
        for (int s = 0; s < n_skus; ++s)
          covered_shortage += std::min(pack[s], std::max(0, shortage[s]));
        if (covered_shortage <= 0)
          continue;

        double delta =
            delta_add_one_pack(pack, shipped_col, demand_col, data.unit_cost);
        double score = delta - rand01(rng) * 0.01 * covered_shortage;
        best.push_back({score, delta, p_idx});
      }
      if (best.empty())
        break;
      std::sort(best.begin(), best.end(), [](const auto &a, const auto &b) {
        return std::get<0>(a) < std::get<0>(b);
      });
      int shortlist = std::min(5, static_cast<int>(best.size()));
      int pick = randint(rng, 0, shortlist - 1);
      double delta = std::get<1>(best[pick]);
      int chosen_p = std::get<2>(best[pick]);
      if (delta >= 0.0)
        break;

      sol.alloc[chosen_p][c_idx] += 1;
      for (int s = 0; s < n_skus; ++s)
        shipped[s][c_idx] += sol.content[chosen_p][s];
      steps++;
    }
  }

  for (int pass = 0; pass < PRUNE_PASSES; ++pass) {
    bool improved = false;
    shipped = shipped_matrix(sol, data);
    std::vector<std::pair<int, int>> pairs;
    for (int p = 0; p < static_cast<int>(sol.content.size()); ++p) {
      for (int c = 0; c < n_channels; ++c) {
        if (sol.alloc[p][c] > 0)
          pairs.push_back({p, c});
      }
    }
    std::shuffle(pairs.begin(), pairs.end(), rng);

    for (auto [p_idx, c_idx] : pairs) {
      while (sol.alloc[p_idx][c_idx] > 0) {
        std::vector<int> shipped_col(n_skus), demand_col(n_skus);
        for (int s = 0; s < n_skus; ++s) {
          shipped_col[s] = shipped[s][c_idx];
          demand_col[s] = data.demand[s][c_idx];
        }
        double delta = delta_remove_one_pack(sol.content[p_idx], shipped_col,
                                             demand_col, data.unit_cost);
        if (delta < -1e-9) {
          sol.alloc[p_idx][c_idx]--;
          for (int s = 0; s < n_skus; ++s)
            shipped[s][c_idx] -= sol.content[p_idx][s];
          improved = true;
        } else {
          break;
        }
      }
    }
    if (!improved)
      break;
  }

  sol = remove_empty_pack_rows(sol);
  evaluate(sol, data);
  return sol;
}

static Solution greedy_construct_solution(ProblemData &data, std::mt19937 &rng,
                                          int n_seed_packs) {
  Solution sol = create_single_sku_initial_solution(data);
  for (int i = 0; i < n_seed_packs; ++i) {
    auto [pack, name] = create_random_curve_pack(data, rng);
    sol = append_pack(std::move(sol), std::move(pack), name);
  }
  sol = repair_allocation(std::move(sol), data, rng);
  evaluate(sol, data);
  return sol;
}

// =============================================================================
// Neighborhood operators
// =============================================================================

static Solution op_add_curve_pack(const Solution &sol, const ProblemData &data,
                                  std::mt19937 &rng) {
  Solution next = sol;
  auto [pack, name] = create_random_curve_pack(data, rng);
  next = append_pack(std::move(next), std::move(pack), name);
  return repair_allocation(std::move(next), data, rng);
}

static Solution op_remove_pack(const Solution &sol, const ProblemData &data,
                               std::mt19937 &rng) {
  if (sol.content.size() <= 1)
    return sol;
  Solution next = sol;
  std::vector<long long> usage(next.alloc.size(), 0);
  for (std::size_t p = 0; p < next.alloc.size(); ++p)
    usage[p] = sum_vector(next.alloc[p]);

  int idx = 0;
  if (rand01(rng) < 0.70) {
    std::vector<int> order(usage.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return usage[a] < usage[b]; });
    int candidate_count = std::max(1, static_cast<int>(usage.size()) / 4);
    idx = order[randint(rng, 0, candidate_count - 1)];
  } else {
    idx = randint(rng, 0, static_cast<int>(next.content.size()) - 1);
  }

  next.content.erase(next.content.begin() + idx);
  next.alloc.erase(next.alloc.begin() + idx);
  next.names.erase(next.names.begin() + idx);
  return repair_allocation(std::move(next), data, rng);
}

static Solution op_mutate_pack_content(const Solution &sol,
                                       const ProblemData &data,
                                       std::mt19937 &rng) {
  if (sol.content.empty())
    return sol;
  Solution next = sol;
  int p = randint(rng, 0, static_cast<int>(next.content.size()) - 1);
  std::vector<int> pack = next.content[p];
  int n_skus = static_cast<int>(data.sku_ids.size());

  std::vector<int> demand_total(n_skus, 0);
  std::vector<int> positive_skus;
  for (int s = 0; s < n_skus; ++s) {
    demand_total[s] = static_cast<int>(sum_vector(data.demand[s]));
    if (demand_total[s] > 0)
      positive_skus.push_back(s);
  }
  if (positive_skus.empty())
    return sol;

  std::vector<std::string> actions = {"add", "remove", "increment", "decrement",
                                      "swap"};
  std::string action =
      actions[randint(rng, 0, static_cast<int>(actions.size()) - 1)];

  if (action == "add") {
    if (count_nonzero(pack) < MAX_DISTINCT_SKUS_PER_PACK &&
        sum_vector(pack) < MAX_PACK_UNITS) {
      int s = positive_skus[randint(
          rng, 0, static_cast<int>(positive_skus.size()) - 1)];
      std::vector<int> existing;
      for (int i = 0; i < n_skus; ++i)
        if (pack[i] > 0)
          existing.push_back(i);
      if (!existing.empty() && rand01(rng) < 0.60) {
        int ref =
            existing[randint(rng, 0, static_cast<int>(existing.size()) - 1)];
        std::string ref_product = data.sku_meta[ref].product_id;
        std::vector<int> related;
        for (int i = 0; i < n_skus; ++i) {
          if (data.sku_meta[i].product_id == ref_product &&
              demand_total[i] > 0 && pack[i] == 0)
            related.push_back(i);
        }
        if (!related.empty())
          s = related[randint(rng, 0, static_cast<int>(related.size()) - 1)];
      } else {
        std::vector<double> weights;
        for (int idx : positive_skus)
          weights.push_back(static_cast<double>(demand_total[idx]));
        s = positive_skus[random_index_weighted(rng, weights)];
      }
      pack[s] += 1;
    }
  } else if (action == "remove") {
    std::vector<int> nonzero;
    for (int i = 0; i < n_skus; ++i)
      if (pack[i] > 0)
        nonzero.push_back(i);
    if (nonzero.size() > 1)
      pack[nonzero[randint(rng, 0, static_cast<int>(nonzero.size()) - 1)]] = 0;
  } else if (action == "increment") {
    std::vector<int> nonzero;
    for (int i = 0; i < n_skus; ++i)
      if (pack[i] > 0)
        nonzero.push_back(i);
    if (!nonzero.empty() && sum_vector(pack) < MAX_PACK_UNITS)
      pack[nonzero[randint(rng, 0, static_cast<int>(nonzero.size()) - 1)]]++;
  } else if (action == "decrement") {
    std::vector<int> nonzero;
    for (int i = 0; i < n_skus; ++i)
      if (pack[i] > 0)
        nonzero.push_back(i);
    if (!nonzero.empty()) {
      int s = nonzero[randint(rng, 0, static_cast<int>(nonzero.size()) - 1)];
      pack[s] = std::max(0, pack[s] - 1);
    }
  } else if (action == "swap") {
    std::vector<int> nonzero;
    std::vector<int> zero_positive;
    for (int i = 0; i < n_skus; ++i) {
      if (pack[i] > 0)
        nonzero.push_back(i);
      if (demand_total[i] > 0 && pack[i] == 0)
        zero_positive.push_back(i);
    }
    if (!nonzero.empty() && !zero_positive.empty()) {
      int s_out =
          nonzero[randint(rng, 0, static_cast<int>(nonzero.size()) - 1)];
      int s_in = zero_positive[randint(
          rng, 0, static_cast<int>(zero_positive.size()) - 1)];
      int qty = pack[s_out];
      pack[s_out] = 0;
      pack[s_in] = std::max(1, qty);
    }
  }

  if (sum_vector(pack) <= 0)
    return sol;
  while (sum_vector(pack) > MAX_PACK_UNITS) {
    auto it = std::max_element(pack.begin(), pack.end());
    if (it == pack.end() || *it <= 0)
      break;
    (*it)--;
  }

  next.content[p] = std::move(pack);
  next.names[p] = "mut_" + next.names[p];
  return repair_allocation(std::move(next), data, rng);
}

static Solution op_merge_packs(const Solution &sol, const ProblemData &data,
                               std::mt19937 &rng) {
  if (sol.content.size() < 2)
    return sol;
  Solution next = sol;
  int p1 = randint(rng, 0, static_cast<int>(next.content.size()) - 1);
  int p2 = randint(rng, 0, static_cast<int>(next.content.size()) - 2);
  if (p2 >= p1)
    p2++;

  std::vector<int> pack = next.content[p1];
  for (std::size_t s = 0; s < pack.size(); ++s)
    pack[s] += next.content[p2][s];
  while (sum_vector(pack) > MAX_PACK_UNITS) {
    auto it = std::max_element(pack.begin(), pack.end());
    if (it == pack.end() || *it <= 0)
      break;
    (*it)--;
  }

  next = append_pack(std::move(next), std::move(pack),
                     "merge_" + std::to_string(p1) + "_" + std::to_string(p2));

  if (rand01(rng) < 0.50 && next.content.size() > 2) {
    int remove_idx = (rand01(rng) < 0.5) ? p1 : p2;
    next.content.erase(next.content.begin() + remove_idx);
    next.alloc.erase(next.alloc.begin() + remove_idx);
    next.names.erase(next.names.begin() + remove_idx);
  }
  return repair_allocation(std::move(next), data, rng);
}

static Solution op_split_pack(const Solution &sol, const ProblemData &data,
                              std::mt19937 &rng) {
  if (sol.content.empty())
    return sol;
  Solution next = sol;
  std::vector<int> candidates;
  for (int p = 0; p < static_cast<int>(next.content.size()); ++p)
    if (sum_vector(next.content[p]) >= 2)
      candidates.push_back(p);
  if (candidates.empty())
    return sol;

  int p = candidates[randint(rng, 0, static_cast<int>(candidates.size()) - 1)];
  std::vector<int> pack = next.content[p];
  std::vector<int> nonzero;
  for (int i = 0; i < static_cast<int>(pack.size()); ++i)
    if (pack[i] > 0)
      nonzero.push_back(i);

  std::vector<int> pack1(pack.size(), 0), pack2(pack.size(), 0);
  if (nonzero.size() <= 1) {
    int s = nonzero[0];
    int q1 = std::max(1, pack[s] / 2);
    int q2 = pack[s] - q1;
    if (q2 <= 0)
      return sol;
    pack1[s] = q1;
    pack2[s] = q2;
  } else {
    std::shuffle(nonzero.begin(), nonzero.end(), rng);
    int cut = randint(rng, 1, static_cast<int>(nonzero.size()) - 1);
    for (int i = 0; i < static_cast<int>(nonzero.size()); ++i) {
      if (i < cut)
        pack1[nonzero[i]] = pack[nonzero[i]];
      else
        pack2[nonzero[i]] = pack[nonzero[i]];
    }
  }

  std::vector<int> old_alloc = next.alloc[p];
  next.content.erase(next.content.begin() + p);
  next.alloc.erase(next.alloc.begin() + p);
  next.names.erase(next.names.begin() + p);

  next = append_pack(std::move(next), std::move(pack1),
                     "split_a_" + std::to_string(p));
  next.alloc.back() = old_alloc;
  next = append_pack(std::move(next), std::move(pack2),
                     "split_b_" + std::to_string(p));
  next.alloc.back() = old_alloc;
  return repair_allocation(std::move(next), data, rng);
}

static Solution op_reallocate(const Solution &sol, const ProblemData &data,
                              std::mt19937 &rng) {
  if (sol.content.empty())
    return sol;
  Solution next = sol;
  int moves = randint(rng, 1, 10);
  for (int i = 0; i < moves; ++i) {
    int p = randint(rng, 0, static_cast<int>(next.content.size()) - 1);
    int c = randint(rng, 0, static_cast<int>(next.alloc[p].size()) - 1);
    if (rand01(rng) < 0.50 && next.alloc[p][c] > 0) {
      int dec = randint(rng, 1, std::min(3, next.alloc[p][c]));
      next.alloc[p][c] -= dec;
    } else {
      next.alloc[p][c] += randint(rng, 1, 3);
    }
  }
  return repair_allocation(std::move(next), data, rng);
}

using OperatorFn = Solution (*)(const Solution &, const ProblemData &,
                                std::mt19937 &);

struct OperatorDef {
  std::string name;
  OperatorFn fn;
  double base_weight;
};

static const std::vector<OperatorDef> OPERATORS = {
    {"add_curve_pack", op_add_curve_pack, 1.30},
    {"remove_pack", op_remove_pack, 1.00},
    {"mutate_pack_content", op_mutate_pack_content, 1.40},
    {"merge_packs", op_merge_packs, 1.10},
    {"split_pack", op_split_pack, 0.70},
    {"reallocate", op_reallocate, 0.90},
};

static std::pair<std::string, OperatorFn>
choose_operator(std::mt19937 &rng,
                const std::unordered_map<std::string, double> &weights) {
  std::vector<double> probs;
  probs.reserve(OPERATORS.size());
  for (const auto &op : OPERATORS)
    probs.push_back(weights.at(op.name));
  int idx = random_index_weighted(rng, probs);
  return {OPERATORS[idx].name, OPERATORS[idx].fn};
}

// =============================================================================
// Metaheuristic optimizer
// =============================================================================

static Solution
enforce_pack_type_limit(Solution sol, const ProblemData &data,
                        std::mt19937 &rng,
                        const std::optional<int> &max_pack_types) {
  if (!max_pack_types.has_value())
    return sol;
  sol = remove_empty_pack_rows(sol);

  while (static_cast<int>(sol.content.size()) > *max_pack_types) {
    evaluate(sol, data);
    std::vector<long long> usage(sol.alloc.size(), 0);
    for (std::size_t p = 0; p < sol.alloc.size(); ++p)
      usage[p] = sum_vector(sol.alloc[p]);
    long long min_usage = *std::min_element(usage.begin(), usage.end());
    std::vector<int> candidates;
    for (int p = 0; p < static_cast<int>(usage.size()); ++p)
      if (usage[p] == min_usage)
        candidates.push_back(p);
    int idx =
        candidates[randint(rng, 0, static_cast<int>(candidates.size()) - 1)];
    sol.content.erase(sol.content.begin() + idx);
    sol.alloc.erase(sol.alloc.begin() + idx);
    sol.names.erase(sol.names.begin() + idx);
    sol = repair_allocation(std::move(sol), data, rng);
  }
  return sol;
}

static std::pair<Solution, std::vector<LogRow>>
optimize_metaheuristic(ProblemData &data, int iterations, int restarts,
                       int seed, double initial_temperature,
                       double cooling_rate, int n_seed_packs,
                       const std::optional<int> &time_limit,
                       const std::optional<int> &max_pack_types) {
  auto start = std::chrono::steady_clock::now();
  std::mt19937 global_rng(seed);

  std::optional<Solution> best_global;
  std::vector<LogRow> log_rows;

  std::unordered_map<std::string, double> base_weights;
  for (const auto &op : OPERATORS)
    base_weights[op.name] = op.base_weight;

  for (int restart = 1; restart <= restarts; ++restart) {
    std::mt19937 rng(static_cast<uint32_t>(randint(global_rng, 1, 1000000000)));

    Solution current = greedy_construct_solution(data, rng, n_seed_packs);
    current =
        enforce_pack_type_limit(std::move(current), data, rng, max_pack_types);
    current = repair_allocation(std::move(current), data, rng);
    evaluate(current, data);

    Solution best_restart = current;
    double temperature = initial_temperature;
    auto operator_weights = base_weights;

    int last_update = -1;
    for (int it = 1; it <= iterations; ++it) {

      int progress = (it * 100) / iterations;
      if (progress % 5 == 0 && progress > last_update) {
        std::cout << progress << "% of iterations completed" << std::endl;
        last_update = progress;
      }

      if (time_limit.has_value() && seconds_since(start) >= *time_limit)
        break;

      auto [op_name, op_fn] = choose_operator(rng, operator_weights);
      Solution candidate = op_fn(current, data, rng);
      candidate = enforce_pack_type_limit(std::move(candidate), data, rng,
                                          max_pack_types);
      candidate = repair_allocation(std::move(candidate), data, rng);
      evaluate(candidate, data);

      double delta = candidate.cost - current.cost;
      bool accepted = false;
      if (delta <= 0.0) {
        accepted = true;
      } else {
        double prob = std::exp(-delta / std::max(temperature, 1e-9));
        accepted = rand01(rng) < prob;
      }
      if (accepted)
        current = candidate;

      if (candidate.cost < best_restart.cost) {
        best_restart = candidate;
        operator_weights[op_name] = operator_weights[op_name] * 1.03;
      }
      if (!best_global.has_value() || best_restart.cost < best_global->cost)
        best_global = best_restart;

      if (it % 250 == 0 || it == 1) {
        LogRow row;
        row.restart = restart;
        row.iteration = it;
        row.op = op_name;
        row.current_cost = current.cost;
        row.restart_best_cost = best_restart.cost;
        row.global_best_cost = best_global
                                   ? best_global->cost
                                   : std::numeric_limits<double>::quiet_NaN();
        row.temperature = temperature;
        row.pack_types_current = static_cast<int>(current.content.size());
        row.pack_types_best = static_cast<int>(best_restart.content.size());
        row.shortage_units_best = best_restart.shortage_units;
        row.overstock_units_best = best_restart.overstock_units;
        row.elapsed_seconds = seconds_since(start);
        log_rows.push_back(std::move(row));
      }

      temperature *= cooling_rate;
    }

    std::cout << "Restart " << std::setw(2) << restart << "/" << restarts
              << ": best cost=" << std::fixed << std::setprecision(2)
              << best_restart.cost << ", packs=" << best_restart.content.size()
              << ", allocated=" << best_restart.allocated_packs
              << ", over=" << best_restart.overstock_units
              << ", short=" << best_restart.shortage_units << "\n";

    if (time_limit.has_value() && seconds_since(start) >= *time_limit) {
      std::cout << "Time limit reached.\n";
      break;
    }
  }

  if (!best_global.has_value())
    throw std::runtime_error("Metaheuristic did not produce a solution.");
  std::mt19937 final_rng(static_cast<uint32_t>(seed + 999));
  Solution best = repair_allocation(*best_global, data, final_rng);
  evaluate(best, data);
  return {best, log_rows};
}

// =============================================================================
// Export
// =============================================================================

static std::vector<int> active_pack_indices(const Solution &sol) {
  std::vector<int> active;
  for (int p = 0; p < static_cast<int>(sol.content.size()); ++p) {
    if (sum_vector(sol.content[p]) > 0 && sum_vector(sol.alloc[p]) > 0)
      active.push_back(p);
  }
  return active;
}

static void write_packs_csv(const fs::path &path, const Solution &sol,
                            const ProblemData &data,
                            const std::vector<int> &active) {
  std::ofstream out(path);
  out << "pack_id,pack_name";
  for (const auto &sku : data.sku_ids)
    out << "," << csv_escape(sku);
  out << "\n";
  for (std::size_t i = 0; i < active.size(); ++i) {
    int p = active[i];
    out << "P" << std::setw(3) << std::setfill('0') << (i + 1)
        << std::setfill(' ') << "," << csv_escape(sol.names[p]);
    for (int q : sol.content[p])
      out << "," << q;
    out << "\n";
  }
}

static void write_allocation_csv(const fs::path &path, const Solution &sol,
                                 const ProblemData &data,
                                 const std::vector<int> &active) {
  std::ofstream out(path);
  out << "pack_id";
  for (const auto &ch : data.channel_ids)
    out << "," << csv_escape(ch);
  out << "\n";
  for (std::size_t i = 0; i < active.size(); ++i) {
    int p = active[i];
    out << "P" << std::setw(3) << std::setfill('0') << (i + 1)
        << std::setfill(' ');
    for (int q : sol.alloc[p])
      out << "," << q;
    out << "\n";
  }
}

static void write_assortment_csv(const fs::path &path, const Solution &sol,
                                 const ProblemData &data) {
  std::ofstream out(path);
  auto shipped = shipped_matrix(sol, data);
  out << "sku_id";
  for (const auto &ch : data.channel_ids)
    out << "," << csv_escape(ch);
  out << "\n";
  for (std::size_t s = 0; s < data.sku_ids.size(); ++s) {
    out << csv_escape(data.sku_ids[s]);
    for (int q : shipped[s])
      out << "," << q;
    out << "\n";
  }
}

static void write_diagnostics_csv(const fs::path &path, const Solution &sol,
                                  const ProblemData &data) {
  std::ofstream out(path);
  auto shipped = shipped_matrix(sol, data);
  out << "sku_id,channel_id,forecast_units,shipped_units,overstock_units,"
         "shortage_units,unit_cost,capital_cost\n";
  for (std::size_t s = 0; s < data.sku_ids.size(); ++s) {
    for (std::size_t c = 0; c < data.channel_ids.size(); ++c) {
      int forecast = data.demand[s][c];
      int sent = shipped[s][c];
      int over = std::max(0, sent - forecast);
      int under = std::max(0, forecast - sent);
      double capital = over * data.unit_cost[s] * COST_OF_CAPITAL;
      out << csv_escape(data.sku_ids[s]) << ","
          << csv_escape(data.channel_ids[c]) << "," << forecast << "," << sent
          << "," << over << "," << under << "," << data.unit_cost[s] << ","
          << capital << "\n";
    }
  }
}

static void write_cost_summary_csv(const fs::path &path, const Solution &sol,
                                   const ProblemData &data,
                                   const std::vector<int> &active) {
  auto shipped = shipped_matrix(sol, data);
  long long total_forecast = 0;
  long long total_shipped = 0;
  long long total_over = 0;
  long long total_under = 0;
  double capital_cost = 0.0;
  for (std::size_t s = 0; s < data.sku_ids.size(); ++s) {
    for (std::size_t c = 0; c < data.channel_ids.size(); ++c) {
      int forecast = data.demand[s][c];
      int sent = shipped[s][c];
      int over = std::max(0, sent - forecast);
      int under = std::max(0, forecast - sent);
      total_forecast += forecast;
      total_shipped += sent;
      total_over += over;
      total_under += under;
      capital_cost += over * data.unit_cost[s] * COST_OF_CAPITAL;
    }
  }
  long long total_allocated_packs = 0;
  for (int p : active)
    total_allocated_packs += sum_vector(sol.alloc[p]);
  long long total_pack_types = static_cast<long long>(active.size());
  double setup_cost = total_pack_types * PACK_CREATION_COST;
  double handling_cost = total_allocated_packs * HANDLING_COST_PER_PACK;
  double shortage_cost = total_under * SHORTAGE_PENALTY_PER_UNIT;
  double total_cost = setup_cost + handling_cost + capital_cost + shortage_cost;
  double baseline_no_packs_cost = total_forecast * HANDLING_COST_PER_PACK;

  std::ofstream out(path);
  out << "metric,value\n";
  out << "total_cost," << total_cost << "\n";
  out << "setup_cost," << setup_cost << "\n";
  out << "handling_cost," << handling_cost << "\n";
  out << "capital_cost," << capital_cost << "\n";
  out << "shortage_cost," << shortage_cost << "\n";
  out << "pack_types_used," << total_pack_types << "\n";
  out << "allocated_packs_total," << total_allocated_packs << "\n";
  out << "forecast_units_total," << total_forecast << "\n";
  out << "shipped_units_total," << total_shipped << "\n";
  out << "overstock_units_total," << total_over << "\n";
  out << "shortage_units_total," << total_under << "\n";
  out << "baseline_no_packs_handling_cost," << baseline_no_packs_cost << "\n";
  out << "estimated_savings_vs_unit_handling_only,"
      << (baseline_no_packs_cost - total_cost) << "\n";
}

static void write_run_log_csv(const fs::path &path,
                              const std::vector<LogRow> &rows) {
  if (rows.empty())
    return;
  std::ofstream out(path);
  out << "restart,iteration,operator,current_cost,restart_best_cost,global_"
         "best_cost,temperature,pack_types_current,pack_types_best,shortage_"
         "units_best,overstock_units_best,elapsed_seconds\n";
  for (const auto &r : rows) {
    out << r.restart << "," << r.iteration << "," << csv_escape(r.op) << ","
        << r.current_cost << "," << r.restart_best_cost << ","
        << r.global_best_cost << "," << r.temperature << ","
        << r.pack_types_current << "," << r.pack_types_best << ","
        << r.shortage_units_best << "," << r.overstock_units_best << ","
        << r.elapsed_seconds << "\n";
  }
}

static void export_solution(const Solution &sol_in, const ProblemData &data,
                            const std::string &output_dir,
                            const std::vector<LogRow> &run_log) {
  fs::path out_dir(output_dir);
  fs::create_directories(out_dir);
  Solution sol = sol_in;
  evaluate(sol, data);
  auto active = active_pack_indices(sol);

  write_packs_csv(out_dir / "optimalisation_packs_metaheuristic.csv", sol, data,
                  active);
  write_allocation_csv(out_dir /
                           "optimalisation_pack_allocation_metaheuristic.csv",
                       sol, data, active);
  write_assortment_csv(out_dir / "optimalisation_assortment_metaheuristic.csv",
                       sol, data);
  write_diagnostics_csv(
      out_dir / "optimalisation_diagnostics_metaheuristic.csv", sol, data);
  write_cost_summary_csv(out_dir /
                             "optimalisation_cost_summary_metaheuristic.csv",
                         sol, data, active);
  write_run_log_csv(out_dir / "optimalisation_run_log.csv", run_log);

  std::cout << "\nCSV outputs written to: " << out_dir.string() << "\n";
}

// =============================================================================
// CLI
// =============================================================================

static void print_usage(const char *exe) {
  std::cout
      << "Usage: " << exe << " [options]\n\n"
      << "Options:\n"
      << "  --forecast PATH              Forecast CSV generated by "
         "run_forecasting.py\n"
      << "  --products PATH              Optional product CSV with id,cost "
         "columns\n"
      << "  --template PATH              Accepted for compatibility; XLSX "
         "filling is not implemented\n"
      << "  --output-dir DIR             Output directory\n"
      << "  --iterations N               Iterations per restart\n"
      << "  --restarts N                 Number of independent restarts\n"
      << "  --seed N                     Random seed\n"
      << "  --initial-temperature X      Initial simulated annealing "
         "temperature\n"
      << "  --cooling-rate X             Temperature multiplier per iteration\n"
      << "  --seed-packs N               Number of demand-curve packs in "
         "initial construction\n"
      << "  --time-limit N               Optional total runtime limit in "
         "seconds\n"
      << "  --max-pack-types N           Optional maximum number of pack "
         "types\n"
      << "  --min-forecast N             Ignore SKUs with total forecast below "
         "this value\n"
      << "  --no-export                  Do not write CSV outputs\n"
      << "  --help                       Show this help\n";
}

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string &k) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("Missing value for " + k);
      return argv[++i];
    };
    if (key == "--help" || key == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (key == "--forecast")
      args.forecast = need_value(key);
    else if (key == "--products")
      args.products = need_value(key);
    else if (key == "--template")
      args.template_path = need_value(key);
    else if (key == "--output-dir")
      args.output_dir = need_value(key);
    else if (key == "--iterations")
      args.iterations = std::stoi(need_value(key));
    else if (key == "--restarts")
      args.restarts = std::stoi(need_value(key));
    else if (key == "--seed")
      args.seed = std::stoi(need_value(key));
    else if (key == "--initial-temperature")
      args.initial_temperature = std::stod(need_value(key));
    else if (key == "--cooling-rate")
      args.cooling_rate = std::stod(need_value(key));
    else if (key == "--seed-packs")
      args.seed_packs = std::stoi(need_value(key));
    else if (key == "--time-limit")
      args.time_limit = std::stoi(need_value(key));
    else if (key == "--max-pack-types")
      args.max_pack_types = std::stoi(need_value(key));
    else if (key == "--min-forecast")
      args.min_forecast = std::stoi(need_value(key));
    else if (key == "--no-export")
      args.export_solution = false;
    else
      throw std::runtime_error("Unknown argument: " + key);
  }
  return args;
}

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    ProblemData data =
        load_problem(args.forecast, args.products, args.min_forecast);

    long long total_forecast = 0;
    for (const auto &row : data.demand)
      total_forecast += sum_vector(row);

    std::cout << std::string(78, '=') << "\n";
    std::cout << "METAHEURISTIC PRE-PACK OPTIMIZATION\n";
    std::cout << std::string(78, '=') << "\n";
    std::cout << "Forecast file        : " << args.forecast << "\n";
    std::cout << "Product file         : " << args.products << "\n";
    std::cout << "SKUs                 : " << data.sku_ids.size() << "\n";
    std::cout << "Sales channels       : " << data.channel_ids.size() << "\n";
    std::cout << "Forecast units total : " << total_forecast << "\n";
    std::cout << "Max pack units       : " << MAX_PACK_UNITS << "\n";
    std::cout << "Max SKUs per pack    : " << MAX_DISTINCT_SKUS_PER_PACK
              << "\n";
    std::cout << "Max pack types       : "
              << (args.max_pack_types ? std::to_string(*args.max_pack_types)
                                      : std::string("unlimited"))
              << "\n";
    std::cout << "Iterations/restarts  : " << args.iterations << " x "
              << args.restarts << "\n";
    std::cout << std::string(78, '=') << "\n";

    auto [best, run_log] = optimize_metaheuristic(
        data, args.iterations, args.restarts, args.seed,
        args.initial_temperature, args.cooling_rate, args.seed_packs,
        args.time_limit, args.max_pack_types);

    if (args.export_solution)
      export_solution(best, data, args.output_dir, run_log);

    std::cout << "\nFinal solution:\n";
    std::cout << "  total cost       : " << std::fixed << std::setprecision(2)
              << best.cost << "\n";
    std::cout << "  setup cost       : " << best.setup_cost << "\n";
    std::cout << "  handling cost    : " << best.handling_cost << "\n";
    std::cout << "  capital cost     : " << best.capital_cost << "\n";
    std::cout << "  shortage cost    : " << best.shortage_cost << "\n";
    std::cout << "  pack types       : " << best.content.size() << "\n";
    std::cout << "  allocated packs  : " << best.allocated_packs << "\n";
    std::cout << "  overstock units  : " << best.overstock_units << "\n";
    std::cout << "  shortage units   : " << best.shortage_units << "\n";
    std::cout << "\nDone.\n";
  } catch (const std::exception &ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
