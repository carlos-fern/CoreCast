#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

struct Point3 {
  float x;
  float y;
  float z;
};

struct Aabb {
  float min_x;
  float min_y;
  float min_z;
  float max_x;
  float max_y;
  float max_z;
};

struct Int3Key {
  int x;
  int y;
  int z;

  bool operator==(const Int3Key& other) const { return x == other.x && y == other.y && z == other.z; }
};

struct Int3Hash {
  std::size_t operator()(const Int3Key& key) const {
    const std::size_t hx = static_cast<std::size_t>(key.x) * 73856093u;
    const std::size_t hy = static_cast<std::size_t>(key.y) * 19349663u;
    const std::size_t hz = static_cast<std::size_t>(key.z) * 83492791u;
    return hx ^ hy ^ hz;
  }
};

struct VoxelGroup {
  Int3Key key{};
  std::vector<Point3> points{};
  Aabb aabb{};
  bool has_aabb = false;
  uint32_t score = 0;
};

Point3 make_point(const float x, const float y, const float z) { return Point3{x, y, z}; }

Point3 add(const Point3& a, const Point3& b) { return make_point(a.x + b.x, a.y + b.y, a.z + b.z); }

Point3 sub(const Point3& a, const Point3& b) { return make_point(a.x - b.x, a.y - b.y, a.z - b.z); }

Point3 mul(const Point3& a, const float s) { return make_point(a.x * s, a.y * s, a.z * s); }

float dot(const Point3& a, const Point3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

float length(const Point3& v) { return std::sqrt(dot(v, v)); }

Point3 normalize(const Point3& v) {
  const float len = length(v);
  if (len <= 1e-6f) {
    return make_point(1.0f, 0.0f, 0.0f);
  }
  return mul(v, 1.0f / len);
}

Point3 random_unit_direction(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int attempt = 0; attempt < 16; ++attempt) {
    const Point3 dir = make_point(dist(rng), dist(rng), dist(rng));
    const float len = length(dir);
    if (len > 1e-4f) {
      return mul(dir, 1.0f / len);
    }
  }
  return make_point(1.0f, 0.0f, 0.0f);
}

bool ray_box_intersection(const Point3& origin, const Point3& direction, const Aabb& box, float* out_t_enter, float* out_t_exit) {
  float tmin = -std::numeric_limits<float>::infinity();
  float tmax = std::numeric_limits<float>::infinity();

  const std::array<float, 3> o = {origin.x, origin.y, origin.z};
  const std::array<float, 3> d = {direction.x, direction.y, direction.z};
  const std::array<float, 3> bmin = {box.min_x, box.min_y, box.min_z};
  const std::array<float, 3> bmax = {box.max_x, box.max_y, box.max_z};

  for (int axis = 0; axis < 3; ++axis) {
    if (std::fabs(d[axis]) < 1e-8f) {
      if (o[axis] < bmin[axis] || o[axis] > bmax[axis]) {
        return false;
      }
      continue;
    }

    float t1 = (bmin[axis] - o[axis]) / d[axis];
    float t2 = (bmax[axis] - o[axis]) / d[axis];
    if (t1 > t2) {
      std::swap(t1, t2);
    }
    tmin = std::max(tmin, t1);
    tmax = std::min(tmax, t2);
    if (tmin > tmax) {
      return false;
    }
  }

  *out_t_enter = tmin;
  *out_t_exit = tmax;
  return true;
}

uint32_t count_points_near_ray_segment(const Point3& origin, const Point3& direction, float t_enter, float t_exit,
                                       float tube_radius, const std::vector<Point3>& all_points) {
  const float radius_sq = tube_radius * tube_radius;
  uint32_t count = 0;

  for (const Point3& p : all_points) {
    const Point3 v = sub(p, origin);
    const float t = dot(v, direction);
    if (t < t_enter || t > t_exit) {
      continue;
    }

    const Point3 closest = add(origin, mul(direction, t));
    const Point3 d = sub(p, closest);
    const float dist_sq = dot(d, d);
    if (dist_sq <= radius_sq) {
      ++count;
    }
  }

  return count;
}

std::vector<Point3> make_demo_point_cloud() {
  std::vector<Point3> points;
  points.reserve(2200);

  // Main structure: near-planar band around z ~= 10.
  for (int yi = -20; yi <= 20; ++yi) {
    for (int xi = -20; xi <= 20; ++xi) {
      const float x = static_cast<float>(xi) * 0.25f;
      const float y = static_cast<float>(yi) * 0.25f;
      const float z = 10.0f + 0.02f * x - 0.01f * y;
      points.push_back(make_point(x, y, z));
    }
  }

  // Outlier cluster #1.
  for (int i = 0; i < 160; ++i) {
    const float t = static_cast<float>(i);
    points.push_back(make_point(12.0f + 0.02f * t, -8.0f + 0.01f * t, 18.0f + 0.03f * t));
  }

  // Outlier cluster #2.
  for (int i = 0; i < 120; ++i) {
    const float t = static_cast<float>(i);
    points.push_back(make_point(-11.0f + 0.03f * t, 7.0f - 0.015f * t, 5.0f + 0.02f * t));
  }

  return points;
}

}  // namespace

int main() {
  // ---------------------------------------------------------------------------
  // Stage 0: Input point cloud.
  // ---------------------------------------------------------------------------
  const std::vector<Point3> input_cloud = make_demo_point_cloud();
  std::cout << "Input cloud size: " << input_cloud.size() << " points\n";

  // ---------------------------------------------------------------------------
  // Stage 1 (coresac.cu equivalent): valid hit point collection.
  // In this CPU demo we treat all points as already-valid hits.
  // ---------------------------------------------------------------------------
  const std::vector<Point3> hit_points = input_cloud;
  std::cout << "[Stage 1] hit_points count: " << hit_points.size() << "\n";

  // ---------------------------------------------------------------------------
  // Stage 2 (coresca_group.cu equivalent): voxel grouping.
  // ---------------------------------------------------------------------------
  const float voxel_size = 1.0f;
  std::unordered_map<Int3Key, VoxelGroup, Int3Hash> voxel_map;
  voxel_map.reserve(hit_points.size() / 8);

  for (const Point3& p : hit_points) {
    const Int3Key key{
        static_cast<int>(std::floor(p.x / voxel_size)),
        static_cast<int>(std::floor(p.y / voxel_size)),
        static_cast<int>(std::floor(p.z / voxel_size)),
    };

    auto [it, inserted] = voxel_map.try_emplace(key);
    if (inserted) {
      it->second.key = key;
    }
    it->second.points.push_back(p);
  }

  std::vector<VoxelGroup> groups;
  groups.reserve(voxel_map.size());
  for (auto& kv : voxel_map) {
    groups.push_back(std::move(kv.second));
  }
  std::cout << "[Stage 2] voxel groups created: " << groups.size() << "\n";

  // ---------------------------------------------------------------------------
  // Stage 3 (coresac_aabb.cu equivalent): per-voxel AABB.
  // ---------------------------------------------------------------------------
  const uint32_t min_points_per_voxel = 12;
  uint32_t active_aabb_count = 0;

  for (VoxelGroup& group : groups) {
    if (group.points.size() < min_points_per_voxel) {
      group.has_aabb = false;
      continue;
    }

    Aabb box{
        +std::numeric_limits<float>::max(),
        +std::numeric_limits<float>::max(),
        +std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };

    for (const Point3& p : group.points) {
      box.min_x = std::min(box.min_x, p.x);
      box.min_y = std::min(box.min_y, p.y);
      box.min_z = std::min(box.min_z, p.z);
      box.max_x = std::max(box.max_x, p.x);
      box.max_y = std::max(box.max_y, p.y);
      box.max_z = std::max(box.max_z, p.z);
    }

    group.aabb = box;
    group.has_aabb = true;
    ++active_aabb_count;
  }

  std::cout << "[Stage 3] active voxel AABBs (>= " << min_points_per_voxel << " points): " << active_aabb_count << "\n";

  // ---------------------------------------------------------------------------
  // Stage 4 (coresac_scoring.cu target behavior): "RANSAC-like" support scoring.
  // For each active AABB:
  //   - pick random seed point in that voxel,
  //   - pick random unit direction,
  //   - clip ray to the voxel AABB using slab intersection,
  //   - count all points near the clipped ray segment (tube radius).
  // ---------------------------------------------------------------------------
  std::mt19937 rng(1337u);
  const float tube_radius = 0.20f;
  uint32_t scored_groups = 0;

  for (VoxelGroup& group : groups) {
    if (!group.has_aabb || group.points.empty()) {
      continue;
    }

    std::uniform_int_distribution<size_t> pick_seed(0, group.points.size() - 1);
    const Point3 origin = group.points[pick_seed(rng)];
    const Point3 direction = random_unit_direction(rng);

    float t_enter = 0.0f;
    float t_exit = 0.0f;
    if (!ray_box_intersection(origin, direction, group.aabb, &t_enter, &t_exit)) {
      group.score = 0;
      continue;
    }

    // Forward segment only (common for beam tracing semantics).
    const float seg_start = std::max(0.0f, t_enter);
    const float seg_end = t_exit;
    if (seg_end <= seg_start) {
      group.score = 0;
      continue;
    }

    group.score = count_points_near_ray_segment(origin, direction, seg_start, seg_end, tube_radius, hit_points);
    ++scored_groups;
  }

  std::cout << "[Stage 4] scored groups: " << scored_groups << "\n";

  // Print top candidates.
  std::vector<const VoxelGroup*> ranked;
  ranked.reserve(groups.size());
  for (const VoxelGroup& g : groups) {
    if (g.has_aabb) {
      ranked.push_back(&g);
    }
  }
  std::sort(ranked.begin(), ranked.end(), [](const VoxelGroup* a, const VoxelGroup* b) { return a->score > b->score; });

  const size_t to_show = std::min<size_t>(8, ranked.size());
  std::cout << "\nTop " << to_show << " voxel candidates by ray-support score:\n";
  for (size_t i = 0; i < to_show; ++i) {
    const VoxelGroup* g = ranked[i];
    std::cout << "  #" << (i + 1) << " key=(" << g->key.x << ", " << g->key.y << ", " << g->key.z
              << ") points=" << g->points.size() << " score=" << g->score << " aabb=[(" << g->aabb.min_x << ", "
              << g->aabb.min_y << ", " << g->aabb.min_z << ") -> (" << g->aabb.max_x << ", " << g->aabb.max_y << ", "
              << g->aabb.max_z << ")]\n";
  }

  std::cout << "\nWhat this tells you right now:\n"
            << "  - You already have Stage 1-3 logic (hit points -> voxel groups -> AABBs).\n"
            << "  - Your next real GPU step is Stage 4 scoring in OptiX with any-hit counting.\n"
            << "  - This example mirrors the math/dataflow so you can validate behavior before CUDA wiring.\n";

  return 0;
}
