/*** 
 * @Author: Lac_Creeper
 * @Date: 2026-03-23 15:25:21 +0800
 * @LastEditTime: 2026-03-23 15:26:29 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightros/src/multi_track/dataset_gen.cpp
 */
#include "flightros/multi_track/dataset_generator.hpp"


using namespace flightros;

int main(int argc, char** argv) {
  DatasetGenerator generator;
  generator.generate();
  return 0;
}