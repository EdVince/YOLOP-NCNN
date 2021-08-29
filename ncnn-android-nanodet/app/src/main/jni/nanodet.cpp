// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include "cpu.h"
#include "layer.h"

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(YoloV5Focus)

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(AAssetManager* mgr, bool use_gpu)
{
    // 把原有的环境清空一下
    yolop.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolop.opt = ncnn::Option();

    yolop.opt.use_fp16_arithmetic = true;
    yolop.opt.use_fp16_packed = true;
    yolop.opt.use_fp16_storage = true;

#if NCNN_VULKAN
    yolop.opt.use_vulkan_compute = use_gpu;
#endif

    yolop.opt.num_threads = ncnn::get_big_cpu_count();
    yolop.opt.blob_allocator = &blob_pool_allocator;
    yolop.opt.workspace_allocator = &workspace_pool_allocator;

    // 加载模型和设置模型对应的一些参数
    yolop.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    int ret = yolop.load_param(mgr, "yolop-opt.param");
    __android_log_print(ANDROID_LOG_DEBUG, "yolop-opt", "param: %d",ret);
    ret = yolop.load_model(mgr, "yolop-opt.bin");
    __android_log_print(ANDROID_LOG_DEBUG, "yolop-opt", "bin: %d",ret);

    return 0;
}

int NanoDet::detect(cv::Mat& rgb)
{
    ncnn::Mat in_pad = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, 640, 640);

    ncnn::Mat da, ll;

    // yolop
    std::vector<Object> objects;
    {
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
        const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = yolop.create_extractor();
        ex.input("input", in_pad);

        std::vector<Object> proposals;
        { // stride 8
            ncnn::Mat out;
            ex.extract("det0", out);

            __android_log_print(ANDROID_LOG_DEBUG, "yolop-opt", "is 8 ?");

            ncnn::Mat anchors(6);
            anchors[0] = 3.f;
            anchors[1] = 9.f;
            anchors[2] = 5.f;
            anchors[3] = 11.f;
            anchors[4] = 4.f;
            anchors[5] = 20.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        { // stride 16
            ncnn::Mat out;
            ex.extract("det1", out);

            ncnn::Mat anchors(6);
            anchors[0] = 7.f;
            anchors[1] = 18.f;
            anchors[2] = 6.f;
            anchors[3] = 39.f;
            anchors[4] = 12.f;
            anchors[5] = 31.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);
            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        { // stride 32
            ncnn::Mat out;
            ex.extract("det2", out);

            ncnn::Mat anchors(6);
            anchors[0] = 19.f;
            anchors[1] = 50.f;
            anchors[2] = 38.f;
            anchors[3] = 81.f;
            anchors[4] = 68.f;
            anchors[5] = 157.f;

            std::vector<Object> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);
            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        {
            ex.extract("da", da);
            ex.extract("ll", ll);
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = objects[i].x;
            float y0 = objects[i].y;
            float x1 = objects[i].x + objects[i].w;
            float y1 = objects[i].y + objects[i].h;

            // clip
            x0 = std::max(std::min(x0, 639.f), 0.f);
            y0 = std::max(std::min(y0, 639.f), 0.f);
            x1 = std::max(std::min(x1, 639.f), 0.f);
            y1 = std::max(std::min(y1, 639.f), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        cv::rectangle(rgb, cv::Rect(obj.x, obj.y, obj.w, obj.h), cv::Scalar(0, 255, 0), 2);
    }
    float* da0 = da.channel(0);
    float* da1 = da.channel(1);
    float* ll0 = ll.channel(0);
    float* ll1 = ll.channel(1);
    uchar* pix = rgb.data;
    for (int i = 0; i < 640 * 640; i++) {
        if ((*da0) < (*da1)) {
            (*(pix + 2)) = 255;
        }
        if ((*ll0) < (*ll1)) {
            (*(pix + 0)) = 255;
        }
        da0++;
        da1++;
        ll0++;
        ll1++;
        pix += 3;
    }

    return 0;
}
