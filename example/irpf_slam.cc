/**
* This file is part of VDO-SLAM.
*
* Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The Australian National University)
* For more information see <https://github.com/halajun/VDO_SLAM>
*
**/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<unistd.h>

#include<opencv2/core/core.hpp>
#include<opencv2/optflow.hpp>

#include<System.h>

using namespace std;
using namespace std::this_thread;
using namespace std::chrono;

void LoadData(const string &strPathToSequence, vector<string> &vstrFilenamesSEM,
              vector<string> &vstrFilenamesRGB, vector<string> &vstrFilenamesDEP, vector<string> &vstrFilenamesFLO,
              vector<double> &vTimestamps, vector<cv::Mat> &vPoseGT);

void LoadMask(const string &strFilenamesMask, cv::Mat &imMask);

/*
 * This function takes a cv::Mat of an image
 * and returns a string representing the bit 
 * size, type, and channels of the image data. 
 * For example, a one channel matrix of unsigned 
 * 16-bit  integers would return "16U_C1".
 * 
 * Parameters:
 *  mat (cv::Mat&): The matrix of image data
 * 
 * Returns:
 *  r (std:string): Returns string of bit size, 
 *        type, and channels of image data.
 */
std::string getMatType(const cv::Mat& mat) {
    int type = mat.type();
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "_C";
    r += std::to_string(chans);
    return r;
}

int main(int argc, char **argv)
{
    // Check that we have the right amount of input arguments
    if(argc != 3)
    {
        cerr << endl << "Usage: ./irpf_slam path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrFilenamesRGB;
    vector<string> vstrFilenamesDEP;
    vector<string> vstrFilenamesSEM;
    vector<string> vstrFilenamesFLO;
    std::vector<cv::Mat> vPoseGT;
    vector<double> vTimestamps;

    LoadData(argv[2], vstrFilenamesSEM, vstrFilenamesRGB, vstrFilenamesDEP, vstrFilenamesFLO,
                  vTimestamps, vPoseGT);

    // Check consistency in the number of images, depth maps, segmentations and flow maps
    int nImages = vstrFilenamesRGB.size()-1;
    if(vstrFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrFilenamesDEP.size()!=vstrFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for depth map." << endl;
        return 1;
    }
    else if(vstrFilenamesSEM.size()!=vstrFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for segmentation." << endl;
        return 1;
    }
    else if(vstrFilenamesFLO.size()!=vstrFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for flow map." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    VDO_SLAM::System SLAM(argv[1],VDO_SLAM::System::RGBD);

    cout << endl << "--------------------------------------------------------------------------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Create a imTraj matrix for directory display
    cv::Mat imTraj(1000, 1000, CV_8UC3, cv::Scalar(255,255,255));

    // Main loop of execution
    cv::Mat imRGB, imD, mTcw_gt;
    for(int ni=0; ni<nImages; ni++)
    {
        cout << endl;
        cout << "=======================================================" << endl;
        cout << "Processing Frame: " << ni << endl;

        // Read imreadmage and depthmap from file
        imRGB = cv::imread(vstrFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD   = cv::imread(vstrFilenamesDEP[ni],CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imD_f, imD_r;

        // Output the type of Disparity Data for debugging
        if (ni == 0){
            cout << "Type of Disparity Image Data: " << getMatType(imD) << endl;
        }

        // For stereo disparity input
        imD.convertTo(imD_f, CV_32F);

        // Load flow matrix
        cv::Mat imFlow = cv::optflow::readOpticalFlow(vstrFilenamesFLO[ni]);

        // Load semantic mask
        cv::Mat imSem(imRGB.rows, imRGB.cols, CV_32SC1);
        LoadMask(vstrFilenamesSEM[ni],imSem);

        double tframe = vTimestamps[ni];
        mTcw_gt = vPoseGT[ni];

        // Object poses in current frame
        vector<vector<float> > vObjPose_gt(0);

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrFilenamesRGB[ni] << endl;
            return 1;
        }

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD_f,imFlow,imSem,mTcw_gt,vObjPose_gt,tframe,imTraj,nImages);

        // Wait a little bit for plot visualization
        //sleep_for(nanoseconds(100000000));
    }

    // Save camera trajectory
    //SLAM.SaveResults("/root/VDO_SLAM/Evaluation/");

    return 0;
}

/**
 * This function loads data from a dataset sequence, and also gets filenames for it.
 * 
 * Parameters:
 *  strPathToSequence: The path to the dataset folder.
 *  vstrFilenamesSEM: Initially empty; vector filled with filenames of all semantic label text files.
 *  vstrFilenamesRGB: Initially empty; vector filled with filenames of all rgb images.
 *  vstrFilenamesDEP: Initially empty; vector filled with filenames of all depth images.
 *  vstrFilenamesFLO: Initially empty; vector filled with filenames of all flow files.
 *  vTimestamps: Initially empty; vector filled with the timestamp values from strPathToSequence/times.txt.
 *  vPoseGT: Initially empty; vector filled with ground truth pose matricies of the camera/robot.
 */
void LoadData(const string &strPathToSequence, vector<string> &vstrFilenamesSEM,
              vector<string> &vstrFilenamesRGB,vector<string> &vstrFilenamesDEP, vector<string> &vstrFilenamesFLO,
              vector<double> &vTimestamps, vector<cv::Mat> &vPoseGT)
{
    // Load the timestamps
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    fTimes.close();

    // +++ image, disparity, semantic and moving object tracking mask +++
    string strPrefixImage = strPathToSequence + "/image_L/";         
    string strPrefixDepth = strPathToSequence + "/disparity_L/";
    string strPrefixSemantic = strPathToSequence + "/semantic_L/";
    string strPrefixFlow = strPathToSequence + "/flow_L/";

    const int nTimes = vTimestamps.size();
    vstrFilenamesRGB.resize(nTimes);
    vstrFilenamesDEP.resize(nTimes);
    vstrFilenamesSEM.resize(nTimes);
    vstrFilenamesFLO.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrFilenamesRGB[i] = strPrefixImage + ss.str() + ".jpeg";
        vstrFilenamesDEP[i] = strPrefixDepth + ss.str() + ".png";
        vstrFilenamesSEM[i] = strPrefixSemantic + ss.str() + ".txt";
        vstrFilenamesFLO[i] = strPrefixFlow + ss.str() + ".flo";
    }


    // +++ ground truth pose +++
    string strFilenamePose = strPathToSequence + "/pose_gt.txt"; //  pose_gt.txt  kevin_extrinsics.txt
    // vPoseGT.resize(nTimes);
    ifstream fPose;
    fPose.open(strFilenamePose.c_str());
    while(!fPose.eof())
    {
        string s;
        getline(fPose,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            int t;
            ss >> t;
            cv::Mat Pose_tmp = cv::Mat::eye(4,4,CV_32F);
            ss >> Pose_tmp.at<float>(0,0) >> Pose_tmp.at<float>(0,1) >> Pose_tmp.at<float>(0,2) >> Pose_tmp.at<float>(0,3)
               >> Pose_tmp.at<float>(1,0) >> Pose_tmp.at<float>(1,1) >> Pose_tmp.at<float>(1,2) >> Pose_tmp.at<float>(1,3)
               >> Pose_tmp.at<float>(2,0) >> Pose_tmp.at<float>(2,1) >> Pose_tmp.at<float>(2,2) >> Pose_tmp.at<float>(2,3)
               >> Pose_tmp.at<float>(3,0) >> Pose_tmp.at<float>(3,1) >> Pose_tmp.at<float>(3,2) >> Pose_tmp.at<float>(3,3);

            cout << Pose_tmp << endl;
            vPoseGT.push_back(Pose_tmp);
            // if(t==410)
            //     cout << "ground truth pose 0 (for validation):" << endl << vPoseGT[t] << endl;
        }
    }
    fPose.close();
}

void LoadMask(const string &strFilenamesMask, cv::Mat &imMask)
{
    ifstream file_mask;
    file_mask.open(strFilenamesMask.c_str());

    // Main loop
    int count = 0;
    cv::Mat imgLabel(imMask.rows,imMask.cols,CV_8UC3); // for display
    while(!file_mask.eof())
    {
        string s;
        getline(file_mask,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            int tmp;
            for(int i = 0; i < imMask.cols; ++i){
                ss >> tmp;
                if (tmp!=0){
                    imMask.at<int>(count,i) = tmp;
                    if (tmp>50)
                        tmp = tmp/2;
                    switch (tmp)
                    {
                        case 0:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0,255,255);
                            break;
                        case 1:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0,0,255);  // red
                            break;
                        case 2:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(255,0,0);  // blue
                            break;
                        case 3:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(255,255,0); // cyan
                            break;
                        case 4:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(47,255,173); // green yellow
                            break;
                        case 5:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(128, 0, 128);
                            break;
                        case 6:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(203,192,255);
                            break;
                        case 7:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(196,228,255);
                            break;
                        case 8:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(42,42,165);
                            break;
                        case 9:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(255,255,255);
                            break;
                        case 10:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(245,245,245); // whitesmoke
                            break;
                        case 11:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0,165,255); // orange
                            break;
                        case 12:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(230,216,173); // lightblue
                            break;
                        case 13:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(128,128,128); // grey
                            break;
                        case 14:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0,215,255); // gold
                            break;
                        case 15:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(30,105,210); // chocolate
                            break;
                        case 16:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0,255,0);  // green
                            break;
                        case 17:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(34, 34, 178);  // firebrick
                            break;
                        case 18:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(240, 255, 240);  // honeydew
                            break;
                        case 19:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(250, 206, 135);  // lightskyblue
                            break;
                        case 20:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                            break;
                        case 21:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(225, 228, 255);  // mistyrose
                            break;
                        case 22:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(128, 0, 0);  // navy
                            break;
                        case 23:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(35, 142, 107);  // olivedrab
                            break;
                        case 24:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(45, 82, 160);  // sienna
                            break;
                        case 25:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0, 255, 127); // chartreuse
                            break;
                        case 26:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(139, 0, 0);  // darkblue
                            break;
                        case 27:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(60, 20, 220);  // crimson
                            break;
                        case 28:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0, 0, 139);  // darkred
                            break;
                        case 29:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(211, 0, 148);  // darkviolet
                            break;
                        case 30:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(255, 144, 30);  // dodgerblue
                            break;
                        case 31:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(105, 105, 105);  // dimgray
                            break;
                        case 32:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(180, 105, 255);  // hotpink
                            break;
                        case 33:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(204, 209, 72);  // mediumturquoise
                            break;
                        case 34:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(173, 222, 255);  // navajowhite
                            break;
                        case 35:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(143, 143, 188); // rosybrown
                            break;
                        case 36:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(50, 205, 50);  // limegreen
                            break;
                        case 37:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(34, 34, 178);  // firebrick
                            break;
                        case 38:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(240, 255, 240);  // honeydew
                            break;
                        case 39:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(250, 206, 135);  // lightskyblue
                            break;
                        case 40:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                            break;
                        case 41:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(225, 228, 255);  // mistyrose
                            break;
                        case 42:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(128, 0, 0);  // navy
                            break;
                        case 43:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(35, 142, 107);  // olivedrab
                            break;
                        case 44:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(45, 82, 160);  // sienna
                            break;
                        case 45:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(30,105,210); // chocolate
                            break;
                        case 46:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(0,255,0);  // green
                            break;
                        case 47:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(34, 34, 178);  // firebrick
                            break;
                        case 48:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(240, 255, 240);  // honeydew
                            break;
                        case 49:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(250, 206, 135);  // lightskyblue
                            break;
                        case 50:
                            imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(238, 104, 123);  // mediumslateblue
                            break;
                    }
                }
                else{
                    imMask.at<int>(count,i) = 0;
                    imgLabel.at<cv::Vec3b>(count,i) = cv::Vec3b(255,255,240); // azure
                }
                // cout << imMask.at<int>(count,i) << " ";
            }
            // cout << endl;
            count++;
        }
    }

    // Display the Instance Segmentation
    cv::imshow("Segmentation Mask", imgLabel);
    cv::waitKey(1);
    return;
}






