/***
 * @Author: Lac_Creeper
 * @Date: 2025-06-03 17:07:44 +0800
 * @LastEditTime: 2025-06-06 22:47:57 +0800
 * @LastEditors: Lac_Creeper
 * @Description:
 * @FilePath: /flightmare/flightlib/src/sensors/lidar2D.cpp
 */
#include "flightlib/sensors/lidar2D.hpp"

namespace flightlib {


Rectangle::Rectangle(Vector<2> center, Scalar width, Scalar height,
                     Scalar angle)
  : center_(center), width_(width), height_(height) {
  rotation_ = Eigen::Rotation2Df(angle).toRotationMatrix();
  inv_rotation_ = rotation_.transpose();
  calculateVertices();
}

void Rectangle::calculateVertices() {
  const Scalar hw = width_ / 2.0;
  const Scalar hh = height_ / 2.0;

  vertices_ = {center_ + rotation_ * Vector<2>(-hw, -hh),
               center_ + rotation_ * Vector<2>(hw, -hh),
               center_ + rotation_ * Vector<2>(hw, hh),
               center_ + rotation_ * Vector<2>(-hw, hh)};
}

AlignedBox2f Rectangle::getBBox() const {
  AlignedBox2f box;
  for (const auto& v : vertices_) {
    box.extend(v);
  }
  return box;
}

bool Rectangle::rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                             Scalar maxDist, Scalar& t) const {
  Vector<2> localOrigin = inv_rotation_ * (origin - center_);
  Vector<2> localDir = inv_rotation_ * dir;

  Scalar tMin = -std::numeric_limits<Scalar>::max();
  Scalar tMax = std::numeric_limits<Scalar>::max();

  const Scalar hw = width_ / 2.0;
  const Scalar hh = height_ / 2.0;

  /// check X axis
  if (abs(localDir.x()) < EPSILON) {
    if (localOrigin.x() < -hw || localOrigin.x() > hw) return false;
  } else {
    Scalar invDx = 1.0 / localDir.x();
    Scalar t1 = (-hw - localOrigin.x()) * invDx;
    Scalar t2 = (hw - localOrigin.x()) * invDx;

    if (t1 > t2) std::swap(t1, t2);
    tMin = std::max(tMin, t1);
    tMax = std::min(tMax, t2);
    if (tMin > tMax) return false;
  }
  /// check Y axis
  if (std::abs(localDir.y()) < EPSILON) {
    if (localOrigin.y() < -hh || localOrigin.y() > hh) return false;
  } else {
    Scalar invDy = 1.0 / localDir.y();
    Scalar t1 = (-hh - localOrigin.y()) * invDy;
    Scalar t2 = (hh - localOrigin.y()) * invDy;

    if (t1 > t2) std::swap(t1, t2);
    tMin = std::max(tMin, t1);
    tMax = std::min(tMax, t2);
    if (tMin > tMax) return false;
  }

  ///
  if (tMin > maxDist || tMax < 0) return false;

  t = (tMin > 0) ? tMin : tMax;
  return t >= 0 && t <= maxDist;
}

Scalar Rectangle::distanceToPoint(const Vector<2>& point) const {
  /// transform to local frame
  Vector<2> localPoint = inv_rotation_ * (point - center_);

  /// calculate the closest point
  Vector<2> closest(std::clamp(localPoint.x(), -width_ / 2, width_ / 2),
                    std::clamp(localPoint.y(), -height_ / 2, height_ / 2));

  return (localPoint - closest).norm();
}

bool Rectangle::containsPoint(const Vector<2>& point) const {
  Vector<2> localPoint = inv_rotation_ * (point - center_);

  return (localPoint.x() >= -width_ / 2) && (localPoint.x() <= width_ / 2) &&
         (localPoint.y() >= -height_ / 2) && (localPoint.y() <= height_ / 2);
}

Ellipse::Ellipse(Vector<2> center, Scalar a, Scalar b, Scalar angle)
  : center_(center), a_{a}, b_{b} {
  rotation_ = Eigen::Rotation2Df(angle).toRotationMatrix();
  inv_rotation_ = rotation_.transpose();
}

AlignedBox2f Ellipse::getBBox() const {
  Vector<2> axis1 = rotation_ * Vector<2>(a_, 0.0f);
  Vector<2> axis2 = rotation_ * Vector<2>(0.0f, b_);

  Scalar halfWidth = std::max(abs(axis1.x()), abs(axis2.x()));
  Scalar halfHeight = std::max(abs(axis1.y()), abs(axis2.y()));

  return AlignedBox2f(
    center_.array() - Vector<2>(halfWidth, halfHeight).array(),
    center_.array() + Vector<2>(halfWidth, halfHeight).array());
}

bool Ellipse::rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                           Scalar maxDist, Scalar& t) const {
  Vector<2> localOrigin = inv_rotation_ * (origin - center_);
  Vector<2> localDir = inv_rotation_ * dir;

  Scalar A = (localDir.x() * localDir.x()) / (a_ * a_) +
             (localDir.y() * localDir.y()) / (b_ * b_);

  Scalar B = 2 * (localOrigin.x() * localDir.x() / (a_ * a_) +
                  localOrigin.y() * localDir.y() / (b_ * b_));

  Scalar C = (localOrigin.x() * localOrigin.x()) / (a_ * a_) +
             (localOrigin.y() * localOrigin.y()) / (b_ * b_) - 1.0;

  Scalar discriminant = B * B - 4 * A * C;
  if (discriminant < 0) return false;

  Scalar sqrtD = sqrt(discriminant);
  Scalar t1 = (-B - sqrtD) / (2 * A);
  Scalar t2 = (-B + sqrtD) / (2 * A);

  if (t1 >= 0 && t1 <= maxDist && t1 <= t2) {
    t = t1;
    return true;
  }
  if (t2 >= 0 && t2 <= maxDist) {
    t = t2;
    return true;
  }
  return false;
}

Scalar Ellipse::distanceToPoint(const Vector<2>& point) const {
  Vector<2> localPoint = inv_rotation_ * (point - center_);

  Scalar r = localPoint.norm();
  if (r < EPSILON) return 0.0;

  Vector<2> dir = localPoint.normalized();
  Vector<2> ellipsePoint =
    dir * a_ * b_ / sqrt(pow(b_ * dir.x(), 2) + pow(a_ * dir.y(), 2));

  return (localPoint - ellipsePoint).norm();
}

bool Ellipse::containsPoint(const Vector<2>& point) const {
  Vector<2> localPoint = inv_rotation_ * (point - center_);

  Scalar normalized = (localPoint.x() * localPoint.x()) / (a_ * a_) +
                      (localPoint.y() * localPoint.y()) / (b_ * b_);
  return normalized <= 1.0;
}


void AABBTree::buildTree(std::vector<std::shared_ptr<Obstacle>>& obstacles,
                         std::unique_ptr<Node>& node, int start, int end) {
  if (start == end) {
    node = std::make_unique<Node>(obstacles[start]);
    return;
  }

  AlignedBox2f overallBox;
  for (int i = start; i < end; ++i) {
    overallBox.extend(obstacles[i]->getBBox());
  }

  Vector<2> extents = overallBox.max() - overallBox.min();
  int axis = (extents.x() > extents.y()) ? 0 : 1;

  int mid = (start + end) / 2;
  std::nth_element(obstacles.begin() + start, obstacles.begin() + mid,
                   obstacles.begin() + end,
                   [axis](const auto& a, const auto& b) {
                     Vector<2> centerA = a->getBBox().center();
                     Vector<2> centerB = b->getBBox().center();
                     return centerA[axis] < centerB[axis];
                   });

  node = std::make_unique<Node>(obstacles[mid]);
  node->box = overallBox;

  if (start < mid) {
    buildTree(obstacles, node->left, start, mid);
  }
  if (mid + 1 < end) {
    buildTree(obstacles, node->right, mid + 1, end);
  }
}

bool AABBTree::rayNodeIntersect(const Node* node, const Vector<2>& origin,
                                const Vector<2>& dir, Scalar maxDist,
                                Scalar& t) const {
  if (!node) return false;

  // 检查射线是否与节点包围盒相交
  Scalar tMin = 0.0, tMax = maxDist;
  for (int i = 0; i < 2; i++) {
    Scalar invDir = 1.0 / dir[i];
    Scalar t0 = (node->box.min()[i] - origin[i]) * invDir;
    Scalar t1 = (node->box.max()[i] - origin[i]) * invDir;

    if (invDir < 0.0) std::swap(t0, t1);
    tMin = std::max(tMin, t0);
    tMax = std::min(tMax, t1);

    if (tMax < tMin) return false;
  }

  // 如果是叶节点，检查障碍物
  if (!node->left && !node->right) {
    return node->obstacle->rayIntersect(origin, dir, maxDist, t);
  }

  // 递归检查子节点
  Scalar tLeft = maxDist, tRight = maxDist;
  bool hitLeft =
    rayNodeIntersect(node->left.get(), origin, dir, maxDist, tLeft);
  bool hitRight =
    rayNodeIntersect(node->right.get(), origin, dir, maxDist, tRight);

  if (hitLeft && hitRight) {
    t = std::min(tLeft, tRight);
    return true;
  }
  if (hitLeft) {
    t = tLeft;
    return true;
  }
  if (hitRight) {
    t = tRight;
    return true;
  }

  return false;
}

void AABBTree::build(std::vector<std::shared_ptr<Obstacle>>& obstacles) {
  if (obstacles.empty()) return;
  buildTree(obstacles, root_, 0, obstacles.size());
}

bool AABBTree::rayIntersect(const Vector<2>& origin, const Vector<2>& dir,
                            Scalar maxDist, Scalar& t) const {
  if (!root_) return false;
  return rayNodeIntersect(root_.get(), origin, dir, maxDist, t);
}


Lidar2D::Lidar2D(Scalar mapWidth, Scalar mapHeight, Scalar fov, int numRays,
                 Scalar maxRange, Scalar safe_range)
  : mapBounds_(Vector<2>(0, 0), Vector<2>(mapWidth, mapHeight)),
    fov_(fov),
    numRays_(numRays),
    maxRange_(maxRange),
    safe_range_(safe_range),
    is_collision_(false) {}

void Lidar2D::addRectangle(Vector<2> center, Scalar width, Scalar height,
                           Scalar angle) {
  auto rect = std::make_shared<Rectangle>(center, width, height, angle);
  obstacles_.push_back(rect);
  treeBuilt_ = false;  // 需要重建树
}

void Lidar2D::addEllipse(Vector<2> center, Scalar a, Scalar b, Scalar angle) {
  auto ellipse = std::make_shared<Ellipse>(center, a, b, angle);
  obstacles_.push_back(ellipse);
  treeBuilt_ = false;  // 需要重建树
}

void Lidar2D::buildTree() {
  if (!treeBuilt_) {
    aabbTree_.build(obstacles_);
    logger_.debug("Tree build success..");
    treeBuilt_ = true;
  }
}

void Lidar2D::generateRandomMap(int numRectangles, int numEllipses,
                                int MAX_ATTEMPTS) {
  // reset
  obstacles_.clear();
  // treeBuilt_ = false;


  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> posX(-mapBounds_.max().x() / 2,
                                              mapBounds_.max().x() / 2);
  std::uniform_real_distribution<Scalar> posY(-mapBounds_.max().y() / 2,
                                              mapBounds_.max().y() / 2);
  std::uniform_real_distribution<Scalar> size(0.3, 0.6);
  std::uniform_real_distribution<Scalar> angle(0, M_PI);

  // 添加随机矩形
  for (int i = 0; i < numRectangles; ++i) {
    bool is_placed = false;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      Scalar cx = posX(gen);
      Scalar cy = posY(gen);
      Scalar width = size(gen);
      Scalar height = size(gen);
      Scalar yaw = angle(gen);
      auto rect =
        std::make_shared<Rectangle>(Vector<2>{cx, cy}, width, height, yaw);
      bool collision = false;
      for (const auto& existing : obstacles_) {
        if (rect->getBBox().intersects(existing->getBBox())) {
          collision = true;
          break;
        }
      }
      if (!collision) {
        obstacles_.push_back(rect);
        treeBuilt_ = false;  // 需要重建树
        is_placed = true;
        break;
      }
    }
    if (!is_placed) std::cerr << "无法放置长方体 " << i << std::endl;
  }

  // 添加随机椭圆
  for (int i = 0; i < numEllipses; ++i) {
    bool is_placed = false;
    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
      Scalar cx = posX(gen);
      Scalar cy = posY(gen);
      Scalar a = size(gen);
      Scalar b = size(gen);
      if (b > a) std::swap(a, b);  // 确保a是长轴
      Scalar yaw = angle(gen);
      auto ellipse = std::make_shared<Ellipse>(Vector<2>{cx, cy}, a, b, yaw);
      bool collision = false;
      for (const auto& existing : obstacles_) {
        if (ellipse->getBBox().intersects(existing->getBBox())) {
          collision = true;
          break;
        }
      }
      if (!collision) {
        obstacles_.push_back(ellipse);
        treeBuilt_ = false;  // 需要重建树
        is_placed = true;
        break;
      }
    }
    if (!is_placed) std::cerr << "无法放置圆柱体 " << i << std::endl;
  }

  // buildTree();
}

bool Lidar2D::simulateLidar(const Vector<2>& robotPos, Scalar robotAngle,
                            std::vector<Scalar>& ranges, bool is_norm) {
  // buildTree();  // 确保树已构建

  ranges.assign(numRays_, maxRange_);
  const Scalar angleStep = fov_ / (numRays_ - 1);
  const Scalar startAngle = robotAngle - fov_ / 2;

  Scalar min_scan = maxRange_;
  bool in_obstacles = false;
  bool is_collision = false;

  // =========== Method 1 =================//
  // FIXME - has bug.
  // for (int i = 0; i < numRays_; ++i) {
  //   Scalar angle = startAngle + i * angleStep;
  //   Vector<2> dir(std::cos(angle), std::sin(angle));

  //   Scalar t = maxRange_;
  //   if (aabbTree_.rayIntersect(robotPos, dir, maxRange_, t)) {
  //     ranges[i] = t;
  //   }
  //   ranges[i] = t;
  // }

  // =========== Method 2 =================//
  for (int i = 0; i < numRays_; ++i) {
    Scalar angle = startAngle + i * angleStep;
    Vector<2> dir(std::cos(angle), std::sin(angle));

    for (const auto& obs : obstacles_) {
      Scalar t_obs;
      if (obs->rayIntersect(robotPos, dir, maxRange_, t_obs)) {
        if (t_obs < ranges[i]) ranges[i] = t_obs;
      }
      if (!in_obstacles && obs->containsPoint(robotPos)) {
        in_obstacles = true;
      }
    }
    if (ranges[i] < min_scan) min_scan = ranges[i];
  }
  if (is_norm) {
    for (int i = 0; i < numRays_; ++i) {
      ranges[i] = ranges[i] / maxRange_ - 0.5f;
    }
  }

  if ((min_scan < safe_range_) || in_obstacles) {
    is_collision = true;
    logger_.warn("ENVIRONMENT COLLISION DETECTED!!");
    logger_.debug("close: %.2f;  in: %d", min_scan, in_obstacles);
  }
  is_collision_ = is_collision;
  return is_collision;

  // =========== Method 3 =================//
  // // 并行处理射线
  // std::vector<std::future<void>> futures;
  // futures.reserve(numRays_);
  // for (int i = 0; i < numRays_; ++i) {
  //   futures.push_back(std::async(std::launch::async, [&, i] {
  //     Scalar rayAngle = startAngle + i * angleStep;
  //     Vector<2> dir(std::cos(rayAngle), std::sin(rayAngle));

  //     Scalar t = maxRange_;
  //     if (aabbTree_.rayIntersect(robotPos, dir, maxRange_, t)) {
  //       ranges[i] = t;
  //     }
  //   }));
  // }
  // // 等待所有射线完成
  // for (auto& f : futures) f.wait();
}

bool Lidar2D::simulateLidar(const QuadState& state, std::vector<Scalar>& ranges,
                            bool is_norm) {
  Vector<2> robot_pose{state.p.x(), state.p.y()};
  Vector<3> euler_xyz = state.euler_xyz();
  Scalar robot_yaw = euler_xyz.z();
  return simulateLidar(robot_pose, robot_yaw, ranges, is_norm);
}

const std::vector<std::shared_ptr<Obstacle>>& Lidar2D::getObstacles() const {
  return obstacles_;
}

std::vector<std::shared_ptr<Obstacle>>& Lidar2D::getObstacles() {
  return obstacles_;
}

bool Lidar2D::isCollision() { return is_collision_; }


}  // namespace flightlib
