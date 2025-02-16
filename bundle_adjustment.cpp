#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "ceres/ceres.h"
#include <ceres/rotation.h>
#include <math.h>
#include <cmath>

#include "dataloader_bf.hpp"
#include "bf_preprocessor.hpp"
#include "dataloader_bal.hpp"
#include "mesh_utils.hpp"
#include "solver_ceres.hpp"
#include "src/sfm.hpp"
#include "src/util/logging.hpp"
#include "src/util/command_parser.hpp"

using namespace std;
using namespace cv;

enum DATA_TYPE {
    UNKNOWN = 0,
    BAL = 1,
    BF = 2,
    GENERIC_SFM = 3,
};

const string _BAL = "bal", _BF = "bf", _SFM = "sfm";

DATA_TYPE to_data_type(const string& in){
    if (in.compare(_BAL) == 0) return DATA_TYPE::BAL;
    if (in.compare(_BF) == 0) return DATA_TYPE::BF;
    if (in.compare(_SFM) == 0) return DATA_TYPE::GENERIC_SFM;
    return DATA_TYPE::UNKNOWN;
}

void print_help_message(){
    cout << "Data type is unknown or not given, please use the following: " << endl
        << "./bundle_adjustment --type <'bal','bf'> --path <path> [options]" << endl
        << "\t--type <type>,\t\tone of ['bal','bf'] (case sensitive)" << endl
        << "\t--path <path>,\t\tpath to data-directory (Bundle Fusion) or to problem.txt (BAL)" << endl
        << "\t--outliers <double>,\tfactor (mean distance) of outliers that should be removed when optimizing, default 2.0" << endl
        << "\t--iterations <int>,\tmaximum number of iterations the solver should optimize"
        << "\t--nimg <int>,\t\tnumber of images to load, default 10 (Bundle Fusion only)" << endl
        << "\t--every <int>,\t\tloads every i-th image of the dataset, default 1 aka every image (Bundle Fusion only)" << endl
        << "\t--matches <int>,\tlimit of matches that should be used for triangulation, default 20 (Bundle Fusion only)" << endl
        << "\t--full-comparison,\tif used, all images are compaired against each other, default false (Bundle Fusion only)" << endl
        << "\t--compare-window <int>,\tcompares each image to the n following images, default 1 (Bundle Fusion only)" << endl
        << "\t--use-poses,\t\tif used, instead of pose estimation the given camera poses will be used (Bundle Fusion only)" << endl;
}

int process_bal(InputParser& parser){
    string path = parser.get_option("--path");
    string outname = path;
    if (path.find('/') != path.size()){
        outname = path.substr(path.find_last_of('/') + 1);
    }
    double removal_scaling = 2.0;
    if (parser.has_option("--outliers")){
        removal_scaling = atof(parser.get_option("--outliers").c_str());
    }
    int iterations = 20;
    if (parser.has_option("--iterations")){
        iterations = atoi(parser.get_option("--iterations").c_str());
    }

    BA_problem problem = BA_problem();
    load_bal(path, problem);
    int removed = BARemoveOutliersRelativ(problem, removal_scaling);
    cout << "Removed " << removed << " Outliers before optimization" << endl;
    balToMesh(problem, outname + "_unoptimized.off");

    cout << "Solving the Bundle Adjustment Problem now..." << endl;
    solveBA(problem, iterations);
    cout << "Successfully solved the Bundle Adjustment Problem" << endl;

    removed = BARemoveOutliersRelativ(problem, removal_scaling * 0.9);
    cout << "Removed " << removed << " Outliers after optimization" << endl;
    balToMesh(problem, outname + "_optimized.off", false, false);
    return 0;
}

int process_bf(InputParser& parser){
    string path = parser.get_option("--path");
    string outname = path;
    if (path.find('/') != path.size()){
        outname = path.substr(path.find_last_of('/') + 1);
    }
    int img_count = 20;
    if (parser.has_option("--nimg")){
        img_count = atoi(parser.get_option("--nimg").c_str());
    }
    int load_every = 1;
    if (parser.has_option("--every")){
        load_every = atoi(parser.get_option("--every").c_str());
    }
    int max_matches = 20;
    if (parser.has_option("--matches")){
        max_matches = atoi(parser.get_option("--matches").c_str());
    }
    bool full_comparison = false;
    if (parser.has_option("--full-comparison")){
        full_comparison = true;
    }
    int comparison_window = 1;
    if (parser.has_option("--compare-window")){
        comparison_window = atoi(parser.get_option("--compare-window").c_str());
    }
    bool use_ref_extrinsics = false;
    if (parser.has_option("--use-poses")){
        use_ref_extrinsics = true;
    }
    double removal_scaling = 2.0;
    if (parser.has_option("--outliers")){
        removal_scaling = atof(parser.get_option("--outliers").c_str());
    }
    int iterations = 20;
    if (parser.has_option("--iterations")){
        iterations = atoi(parser.get_option("--iterations").c_str());
    }

    DataloaderBF loader = DataloaderBF();
    loader.loadImages(path, img_count, load_every);

    SFM_params params = SFM_params{
        .images = loader.imagesColor,
        .focal = (float) loader.info.calibrationColorIntrinsic[0] * 1.0F,
        .width = loader.info.colorWidth,
        .height = loader.info.colorHeight,
        .full_comparison = full_comparison,
        .comparison_window = comparison_window,
        .use_ref_extrinsics = use_ref_extrinsics,
        .ref_extrinsics = loader.cameraPose,
        .match_limit = max_matches,
    };
    BA_problem problem = sfm_pipeline(params);

    cout << "Successfully processed Bundle Fusion set " << path << " with " << problem.num_observations << " Observations" << endl;
    save_bal(problem, outname + "_problem.txt");

    int removed = BARemoveOutliersRelativ(problem, removal_scaling);
    cout << "Removed " << removed << " Outliers before optimization" << endl;
    balToMesh(problem, outname + "_unoptimized.off");

    cout << "Solving the Bundle Adjustment Problem now..." << endl;
    solveBA(problem, iterations);
    cout << "Successfully solved the Bundle Adjustment Problem" << endl;

    removed = BARemoveOutliersRelativ(problem, removal_scaling * 0.9);
    cout << "Removed " << removed << " Outliers after optimization" << endl;
    balToMesh(problem, outname + "_optimized.off");
    return 0;
}

int main ( int argc, char** argv ){
    google::InitGoogleLogging(argv[0]);
    InputParser parser(argc, argv);

    DATA_TYPE data_type = DATA_TYPE::UNKNOWN;
    if (parser.has_option("--type")){
        data_type = to_data_type(parser.get_option("--type"));
    }

    switch (data_type)
    {
    case BAL:
        // do bal only
        cout << "BAL" << endl;
        return process_bal(parser);
    case BF:
        // do sfm on bf dataset
        cout << "BF" << endl;
        return process_bf(parser);
    default:
        // not implemented or unknown
        print_help_message();
        break;
    }

    return 1;
}
