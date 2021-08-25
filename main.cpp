#include <iostream>
#include<chrono>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include"NumCpp.hpp"


const std::string base_dir = "/home/buenos/slam/dataset/myDataset/";
const std::string img_dir = base_dir + "data/";         // img files:  <timestamp>.png
const std::string line_dir = base_dir + "SOLD_data/";   // line files: <timestamp>.pt

class Line
{
public:
    int startPointX;
    int startPointY;
    int endPointX;
    int endPointY;
};

class Frame
{
public:
    Frame(const std::string& timestamp);
    ~Frame(){}
    cv::Mat img;
    std::vector<Line> keylines;
    cv::Mat descriptors;

    cv::Mat GetDescriptors(){ return this->descriptors; }

private:
    std::string timestamp_;    // use for construct img_file and lines_file.
    std::string img_file_, line_file_; 

    // ...
    void LoadImage(const std::string& filename);

    // load lines and line descriptors form <timestamp>.pt
    void LoadLinesAndDescs(const std::string& filename);

    // ...
    void GenDescriptorsFromDT(torch::Tensor& DT);

    void SampleLinePoints(const std::vector<Line>& lines, 
        std::vector<std::vector<double>>& line_points, std::vector<int>& valid_points_num);

    void ExtractPointDescriptors(std::vector<std::vector<double>>& line_points, 
        const torch::Tensor& DT, torch::Tensor& desc);

    void PackLineDescriptors(torch::Tensor& desc, const std::vector<int>& valid_points_num);
};

Frame::Frame(const std::string& timestamp):
timestamp_(timestamp), 
img_file_(img_dir + timestamp + ".png"),
line_file_(line_dir + timestamp + ".pt")
{
    LoadImage(img_file_);
    LoadLinesAndDescs(line_file_);
}

void Frame::LoadImage(const std::string& filename)
{
    img = cv::imread(filename, 0);
    assert(!img.empty());
}

void Frame::LoadLinesAndDescs(const std::string& filename)
{
    torch::jit::script::Module container = torch::jit::load(filename);
    c10::IValue lines = container.attr("line_segments");
    c10::IValue desc = container.attr("descriptor");

    // load lines
    torch::Tensor Lines = lines.toTensor();
    int num_lines = Lines.size(0);
    for(int i = 0; i < num_lines; i++)
    {
        Line line = Line();
        line.startPointX = Lines[i][0][0].item().toInt();
        line.startPointY = Lines[i][0][1].item().toInt();
        line.endPointX   = Lines[i][1][0].item().toInt();
        line.endPointY   = Lines[i][1][1].item().toInt();

        keylines.push_back(line);
    }

    // // TODO: for debug
    // keylines.resize(10);

    // load descriptors
    torch::Tensor DT = desc.toTensor().toType(torch::kDouble);
    GenDescriptorsFromDT(DT);
}

void Frame::GenDescriptorsFromDT(torch::Tensor& DT)
{
    if(DT.sizes().empty())
    {
        return;
    }

    // sampling points along each line
    std::vector<std::vector<double>> line_points;
    std::vector<int> valid_points_num;
    SampleLinePoints(keylines, line_points, valid_points_num);

    // extract super point descriptors
    torch::Tensor desc;
    ExtractPointDescriptors(line_points, DT, desc);

    // construct line descritpors
    PackLineDescriptors(desc, valid_points_num);
}

/* @desc [128, 10 * lines_num]
*  @valid_points_num [lines_num]
*/
void Frame::PackLineDescriptors(torch::Tensor& desc, const std::vector<int>& valid_points_num)
{
    int lines_num = valid_points_num.size();
    descriptors = cv::Mat(lines_num, 128 * 10, CV_64F);

    for(int i = 0; i < lines_num; i++)
    {
        std::vector<torch::Tensor> point_descs;
        for(int j = 0; j < 10; j++)
        {
            if(j < valid_points_num[i])  // copy value from desc
            {
                point_descs.push_back(desc.index({"...", i * 10 + j}));
            } else       // fill with -1
            {
                point_descs.push_back(torch::full({128}, -1.0, torch::dtype(torch::kDouble)));
            }
        }
        torch::Tensor line_desc = torch::cat(point_descs, 0);
        cv::Mat line_desc_mat(1, 128 * 10, CV_64F);
        std::memcpy((void *) line_desc_mat.data, line_desc.data_ptr(), sizeof(double) * line_desc.numel());
        line_desc_mat.row(0).copyTo(descriptors.row(i));
    }
}

void Frame::SampleLinePoints(const std::vector<Line>& lines, std::vector<std::vector<double>>& line_points, std::vector<int>& valid_points_num)
{
    line_points.clear();
    valid_points_num.clear();

    const int num_lines = lines.size();

    const int min_dist_pts = 8;              // Sample the points separated by at least min_dist_pts along each line
    const int max_sample_num = 10;           // Warning: don't change this value, beacuse somewhere use value 10 directly.
    for(Line line: lines)
    {
        int num_valid = 0;

        // sample points
        // step 1: cal line length
        double dx_ = line.endPointX - line.startPointX;
        double dy_ = line.endPointY - line.startPointY;
        double length = std::sqrt(dx_ * dx_ + dy_ * dy_);
        // step 2: cal sample distance
        int smaple_num = std::fmax(2, std::fmin(length / min_dist_pts, max_sample_num));
        double sample_dx = dx_ * 1.0 / (smaple_num - 1);
        double sample_dy = dy_ * 1.0 / (smaple_num - 1);
        for(int i = 0; i < smaple_num; i++)
        {
            line_points.push_back({line.startPointX + sample_dx * i, line.startPointY + sample_dy * i});
            num_valid++;
        }
        valid_points_num.push_back(num_valid);

        // step 3: fill point(0.0, 0.0) to max_sample_num
        for(int i = 0; i < max_sample_num - num_valid; i++)
        {
            line_points.push_back({0.0, 0.0});
        }
    }
}

void Frame::ExtractPointDescriptors(std::vector<std::vector<double>>& line_points, const torch::Tensor& DT, torch::Tensor& desc)
{
    // convert to tensor
    int num_line_points = line_points.size();
    torch::TensorOptions options = torch::TensorOptions().dtype(at::kDouble);
    torch::Tensor line_points_t = torch::zeros({num_line_points, 2}, options);
    for(int i = 0; i < num_line_points; i++)
    {
        line_points_t.slice(0, i, i + 1) = torch::from_blob(line_points[i].data(), {2}, options);
    }

    // cal desc
    torch::Tensor img_size = torch::tensor({img.rows, img.cols}, options);
    torch::Tensor grid = line_points_t * 2. / img_size - 1.;

    torch::Tensor x_slice = grid.index({"...", 0});
    torch::Tensor y_slice = grid.index({"...", 1});
    grid = torch::stack({y_slice, x_slice}, 1);

    torch::Tensor grid_ex = grid.view({1, num_line_points, 1, 2});

    torch::nn::functional::GridSampleFuncOptions opts = torch::nn::functional::GridSampleFuncOptions();
    opts.mode(torch::kBilinear);
    opts.padding_mode(torch::kZeros);
    opts.align_corners(false);
    torch::Tensor sample_res = torch::nn::functional::grid_sample(DT, grid_ex, opts).squeeze();
    desc = torch::nn::functional::normalize(sample_res, torch::nn::functional::NormalizeFuncOptions().dim(0));
}


class Matcher
{
public:
    Matcher(){}
    ~Matcher(){}

    void match(cv::Mat desc1, cv::Mat desc2);

private:
    void UnpackLineDescriptors(cv::Mat& desc, cv::Mat& valid);

    nc::NdArray<int> FilterAndMatchLines(cv::Mat& scores, 
                    const cv::Mat& valid1, const cv::Mat& valid2);

    double NeedlemanWunsch(nc::NdArray<double> nw_scores);

};


void Matcher::match(cv::Mat desc1, cv::Mat desc2) // 1.5 s
{
    int num_lines1 = desc1.rows;

    cv::Mat valid1;
    UnpackLineDescriptors(desc1, valid1);

    cv::Mat valid2;
    UnpackLineDescriptors(desc2, valid2);

    cv::Mat scores = desc1 * desc2.t();  // 0.77 s
    cv::Mat scores2 = scores.t();  // 0.033 s

    
    auto matches = FilterAndMatchLines(scores, valid1, valid2);  // 0.23 s
    auto matches2 = FilterAndMatchLines(scores2, valid2, valid1);
    for(int i = 0; i < num_lines1; i++)
    {
        if(matches2(0, matches(0, i)) != i)
        {
            matches(0, i) = -1;
        }
    }
    // matches.print();

}

nc::NdArray<int> Matcher::FilterAndMatchLines(cv::Mat& scores, 
                    const cv::Mat& valid1, const cv::Mat& valid2)
{
    int rows = scores.rows;
    int cols = scores.cols;
    nc::NdArray<double> nc_scores = nc::zeros<double>(rows, cols);
    for(int i = 0; i < rows; i++)   // 0.10 s
    {
        for(int j = 0; j < cols; j++)
        {
            if(!(valid1.at<double>(0, i) && valid2.at<double>(0, j)))
            {
                scores.at<double>(i,j) = -1;
            } 
            nc_scores(i, j) = scores.at<double>(i,j);
        }
    }


    int num_sample = 10;
    int num_lines1 = rows / num_sample;
    int num_lines2 = cols / num_sample;
    

    cv::Mat line_scores_mat = cv::Mat::zeros(num_lines1, num_lines2, CV_64F);
    for(int i = 0; i < num_lines1; i++)
    {
        for(int j = 0; j < num_lines2; j++)
        {
            cv::Mat line1_score;
            cv::reduce(scores(cv::Range(i * num_sample, i * num_sample + 10), 
                    cv::Range(j * num_sample, j * num_sample + 10)), line1_score, 1, CV_REDUCE_MAX);
            cv::Mat valid1_score = valid1(cv::Range(0, 1), cv::Range(i * 10, i * 10 + 10));
            line1_score = line1_score.t() * valid1_score.t();
            double line1_val = line1_score.at<double>(0, 0) / cv::sum(valid1_score)[0];

            cv::Mat line2_score;
            cv::reduce(scores(cv::Range(i * num_sample, i * num_sample + 10), 
                    cv::Range(j * num_sample, j * num_sample + 10)), line2_score, 0, CV_REDUCE_MAX);
            cv::Mat valid2_score = valid2(cv::Range(0, 1), cv::Range(j * 10, j * 10 + 10));
            line2_score = line2_score * valid2_score.t();
            double line2_val = line2_score.at<double>(0, 0) / cv::sum(valid2_score)[0];
            
            line_scores_mat.at<double>(i, j) = (line1_val + line2_val) * 0.5;
        }
    }

    nc::NdArray<double> line_scores = nc::zeros<double>(num_lines1, num_lines2);
    for(int i = 0; i < num_lines1; i++)
    {
        for(int j = 0; j < num_lines2; j++)
        {
            line_scores(i, j) = line_scores_mat.at<double>(i, j);
        }
    }
    
    auto topk_lines = nc::argsort(line_scores, nc::Axis::COL);
    topk_lines = topk_lines(topk_lines.rSlice(), nc::Slice(std::max(0, num_lines2 - 10), num_lines2));
    
    // top 2k
    int topk = std::min(num_lines2, 10);
    nc::NdArray<double> nw_scores = nc::empty<double>(num_lines1, topk * 2);
    for(int i = 0; i < num_lines1; i++)
    {
        for(int j = 0; j < topk; j++)
        {
            int line2_idx = topk_lines(i, j);
            auto score = nc_scores(nc::Slice(i * num_sample, i * num_sample + 10), nc::Slice(line2_idx * num_sample, line2_idx * num_sample + 10));
            nw_scores(i, j) = NeedlemanWunsch(score);

            score = nc::flip(score, nc::Axis::COL);
            nw_scores(i, topk + j) = NeedlemanWunsch(score);
        }
    }

    auto matches = nc::argmax(nw_scores, nc::Axis::COL);
    for(int i = 0; i < num_lines1; i++)
    {
        int idx = matches(0, i) % topk;
        matches(0, i) = topk_lines(i, idx);
    }
    return matches.astype<int>();
}


double Matcher::NeedlemanWunsch(nc::NdArray<double> nw_scores)
{
    int rows = nw_scores.shape().rows;
    int cols = nw_scores.shape().cols;
    double gap = 0.1;
    nw_scores = nw_scores - gap;

    nc::NdArray<double> nw_grid = nc::zeros<double>(rows + 1, cols + 1);
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            nw_grid(i + 1, j + 1) = std::fmax(
                std::fmax(nw_grid(i + 1, j), nw_grid(i, j + 1)),
                nw_grid(i, j) + nw_scores(i, j)
            );
        }
    }
    return nw_grid(rows, cols);
}


void Matcher::UnpackLineDescriptors(cv::Mat& desc, cv::Mat& valid)
{
    desc = desc.reshape(0, desc.rows * desc.cols / 128);

    // check vaild points
    cv::reduce(desc, valid, 1, CV_REDUCE_SUM);
    valid = valid.t();
    for(int i = 0; i < valid.cols; i++)
    {
        if(valid.at<double>(0, i) < -127.5)
        {
            valid.at<double>(0, i) = 0;
        } else
        {
            valid.at<double>(0, i) = 1;
        }
    }
}

int main()
{
    std::cout << "=============== SOLD2 Impl with C++ ================" << std::endl;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    Frame* frame1 = new Frame("1629107214485519171");
    Frame* frame2 = new Frame("1629107218419335604");
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double newPerFrameTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() / 2.;
    std::cout << "construct Frame cost time: " << newPerFrameTime << std::endl;

    cv::Mat desc1 = frame1->GetDescriptors();
    cv::Mat desc2 = frame2->GetDescriptors();

    Matcher* matcher = new Matcher();
    t1 = std::chrono::steady_clock::now();
    matcher->match(desc1, desc2);
    t2 = std::chrono::steady_clock::now();
    double matchTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    std::cout << "match cost time: " << matchTime << std::endl;

    nc::NdArray<double> a;

    return 0;
}