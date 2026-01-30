/***
 * @Author: Lac_Creeper
 * @Date: 2025-06-03 17:07:25 +0800
 * @LastEditTime: 2025-06-06 22:48:26 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightlib/include/flightlib/sensors/lidar2D.hpp
 */
#pragma once

#include <algorithm>
#include <future>
#include <random>

#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/timer.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/sensors/sensor_base.hpp"

namespace flightlib {

class Obstacle {
 public:
  virtual ~Obstacle() = default;
  virtual AlignedBox2f getBBox() const = 0;
  virtual bool rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                            Scalar maxDist, Scalar& t) const = 0;
  virtual Scalar distanceToPoint(const Vector<2>& point) const = 0;
  virtual bool containsPoint(const Vector<2>& point) const = 0;
};

class Rectangle : public Obstacle {
  Vector<2> center_;
  Scalar width_, height_;
  Matrix<2, 2> rotation_;
  Matrix<2, 2> inv_rotation_;
  std::array<Vector<2>, 4> vertices_;

  void calculateVertices();

 public:
  Rectangle(Vector<2> center, Scalar width, Scalar height, Scalar angle);

  Vector<2> center() { return center_; }
  Scalar width() { return width_; }
  Scalar height() { return height_; }
  Scalar angle() { return acos(rotation_(0)); }

  AlignedBox2f getBBox() const override;

  bool rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                    Scalar maxDist, Scalar& t) const override;

  Scalar distanceToPoint(const Vector<2>& point) const override;

  bool containsPoint(const Vector<2>& point) const override;
};


class Ellipse : public Obstacle {
  Vector<2> center_;
  Scalar a_, b_;
  Matrix<2, 2> rotation_;
  Matrix<2, 2> inv_rotation_;

 public:
  Ellipse(Vector<2> center, Scalar a, Scalar b, Scalar angle);

  Vector<2> center() { return center_; }
  Scalar a() { return a_; }
  Scalar b() { return b_; }
  Scalar angle() { return acos(rotation_(0)); }

  AlignedBox2f getBBox() const override;

  bool rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                    Scalar maxDist, Scalar& t) const override;

  Scalar distanceToPoint(const Vector<2>& point) const override;

  bool containsPoint(const Vector<2>& point) const override;
};

class AABBTree {
  struct Node {
    AlignedBox2f box;
    std::shared_ptr<Obstacle> obstacle;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node(std::shared_ptr<Obstacle> obs) : obstacle(obs), box(obs->getBBox()) {}
  };

  std::unique_ptr<Node> root_;

  void buildTree(std::vector<std::shared_ptr<Obstacle>>& obstacles,
                 std::unique_ptr<Node>& node, int start, int end);

  bool rayNodeIntersect(const Node* node, const Vector<2>& origin,
                        const Vector<2>& dir, Scalar maxDist, Scalar& t) const;

 public:
  void build(std::vector<std::shared_ptr<Obstacle>>& obstacles);

  bool rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                    Scalar maxDist, Scalar& t) const;
};

class Lidar2D {
  std::vector<std::shared_ptr<Obstacle>> obstacles_;
  AABBTree aabbTree_;
  AlignedBox2f mapBounds_;
  bool treeBuilt_ = false;

  Logger logger_{"lidar2d"};
  Logger logger_map_{"", std::string("/home/lac/fm_test/my_logs/random_map.log")};

  /// Lidar parameters
  Scalar fov_;
  int numRays_;
  Scalar maxRange_;
  Scalar safe_range_;
  bool is_collision_;

 public:
  Lidar2D(Scalar mapWidth, Scalar mapHeight, Scalar fov, int numRays,
          Scalar maxRange, Scalar safe_range);

  void addRectangle(Vector<2> center, Scalar width, Scalar height,
                    Scalar angle);

  void addEllipse(Vector<2> center, Scalar a, Scalar b, Scalar angle);

  void buildTree();

  // 生成随机地图
  void generateRandomMap(int numRectangles, int numEllipses,
                         int MAX_ATTEMPTS = 100);
  
  void loadMap(const std::string &filename);

  // 高效的激光雷达模拟
  bool simulateLidar(const Vector<2>& robotPos, Scalar robotAngle,
                     std::vector<Scalar>& ranges, Scalar safeRange,
                     bool is_norm = false);

  bool simulateLidar(const QuadState& state, std::vector<Scalar>& ranges,
                     Scalar safeRange, bool is_norm = false);

  const std::vector<std::shared_ptr<Obstacle>>& getObstacles() const;

  std::vector<std::shared_ptr<Obstacle>>& getObstacles();

  bool isCollision();
};

}  // namespace flightlib