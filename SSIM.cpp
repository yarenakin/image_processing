#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)


double sigma(Mat & m, int i, int j, int block_size)
    {
        double sd = 0;

        Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
        Mat m_squared(block_size, block_size, CV_64F);

        multiply(m_tmp, m_tmp, m_squared);

        // E(x)
        double avg = mean(m_tmp)[0];
        // E(x²)
        double avg_2 = mean(m_squared)[0];


        sd = sqrt(avg_2 - avg * avg);

        return sd;
}

double cov(Mat & m1, Mat & m2, int i, int j, int block_size)
    {
        Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
        Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
        Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));


        multiply(m1_tmp, m2_tmp, m3);

        double avg_ro 	= mean(m3)[0]; // E(XY)
        double avg_r 	= mean(m1_tmp)[0]; // E(X)
        double avg_o 	= mean(m2_tmp)[0]; // E(Y)


        double sd_ro = avg_ro - avg_o * avg_r; // E(XY) - E(X)E(Y)

        return sd_ro;
    }
double ssim(Mat & img_src, Mat & img_compressed, int block_size, bool show_progress = false)
    {
        double ssim = 0;

        int nbBlockPerHeight 	= img_src.rows / block_size;
        int nbBlockPerWidth 	= img_src.cols / block_size;

        for (int k = 0; k < nbBlockPerHeight; k++)
        {
            for (int l = 0; l < nbBlockPerWidth; l++)
            {
                int m = k * block_size;
                int n = l * block_size;

                double avg_o 	= mean(img_src(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double avg_r 	= mean(img_compressed(Range(k, k + block_size), Range(l, l + block_size)))[0];
                double sigma_o 	= sigma(img_src, m, n, block_size);
                double sigma_r 	= sigma(img_compressed, m, n, block_size);
                double sigma_ro	= cov(img_src, img_compressed, m, n, block_size);

                ssim += ((2 * avg_o * avg_r + C1) * (2 * sigma_ro + C2)) / ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));

            }
            // Progress
            if (show_progress)
                cout << "\r>>SSIM [" << (int) ((( (double)k) / nbBlockPerHeight) * 100) << "%]";
        }
        ssim /= nbBlockPerHeight * nbBlockPerWidth;

        if (show_progress)
        {
            cout << "\r>>SSIM [100%]" << endl;
            cout << "SSIM : " << ssim << endl;
        }

        return ssim;
    }


int main()
{
        Mat img_src;
        Mat img_src2;
        int block_size=10;

        img_src = imread("C:/Users/LENOVO/Desktop/gul.jpg",IMREAD_GRAYSCALE);
        img_src2 = imread("C:/Users/LENOVO/Desktop/manolya.jpg",IMREAD_GRAYSCALE);
        resize(img_src,img_src,Size(300,300));
        resize(img_src2,img_src2,Size(300,300));

        imshow("Src",img_src);
        imshow("Src2",img_src2);

        img_src.convertTo(img_src, CV_64F);
        img_src2.convertTo(img_src2, CV_64F);

        int height_1 = img_src.rows;
        int height_2 = img_src2.rows;
        int width_1 = img_src.cols;
        int width_2 = img_src2.cols;

        if (height_1 != height_2 || width_1 != width_2)
        {
            cout << "UYARI1: 2 Resmin Boyutlari Ayni Olmali" << endl;
        }

        if (height_1 % block_size != 0 || width_1 % block_size != 0)
        {
            cout 	<< "UYARI2 :Resim Boyutları block_size ile bölünebilir olmalıdır." << endl
                    << "HEIGHT : " 		<< height_1 	<< endl
                    << "WIDTH : " 		<< width_1	<< endl
                    << "BLOCK_SIZE : " 	<< block_size 	<< endl
                    << endl;
        }

        double ssim_val = ssim(img_src, img_src2, block_size);
        cout << "SSIM : " << ssim_val << endl;
        
        if(ssim_val<0.7) cout<<"RESIMLER FARKLI!"<<endl;
        else cout<<"RESIMLER AYNI!"<<endl;
        
        waitKey(0);
 }
