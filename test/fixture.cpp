#include <filesystem>
#include <iostream>
#include <privret/format.hpp>

int main(const int argc, char** argv){
    if (argc != 2){
        std::cerr << "usage: privret_fixture DIR\n";
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    std::filesystem::create_directories(dir);
    privret::write_file(dir / "docs.i8v", privret::encode_table(privret::make_vectors(
        {"d0", "d1", "d2", "d3", "d4", "d5", "d6"},
        {{127, 127, 127, 127}, {1, 2, 3, 4}, {-5, 0, 5, 10}, {20, -20, 20, -20}, {7, 7, 7, 7}, {0, 0, 0, 1}, {7, 7, 7, 7}}
    )));
    privret::write_file(dir / "queries.i8v", privret::encode_table(privret::make_vectors(
        {"q0", "q1", "q2"}, {{-127, -127, -127, -127}, {1, 1, 1, 1}, {3, -1, 4, -1}}
    )));
    return 0;
}
