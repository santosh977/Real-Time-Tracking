#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "math.h"
#include <iostream>
using namespace cv;
using namespace std;

double dist( double a, double b)                    //distance between a and b
{
    return sqrt((a-b)*(a-b));
}
double Gauss(double x, double m, double v)          //Gaussian function with data point x, mean m and variance v*v
{
    double c= 0.39894228/v;					// that constant value comes after paper calculation of 1/sqrt(2*pi)
    double c1 = (-0.5)*((x-m)*(x-m))/(v*v);
    double p = pow(2.71, c1);
    return c*p;
}

int main()
{
    VideoCapture capture;
    capture.open("umcp.mpg");                           //opening video file

    if(!capture.isOpened())
    {
        cout<<"ERROR ACQUIRING VIDEO FEED\n";           //if file is not opened
        getchar();
        return -1;
    }

    int K=3;
    int col=capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int row=capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int N =	400;							// here we are taking first 400 frames to reduce computation. we can vary this upto 999.
											// For this video capture.get(frame count command) is giving 32 so we are taking N manually.

    double ****frame = new double*** [row];                 //4-D Matrix for N frames of Video with 3 colors BGR
    for (int i = 0; i < row; i++)
    {
        frame[i] = new double**[col];
        for(int j = 0; j< col; j++)
        {
            frame[i][j] = new double*[N];
            for(int k =0; k <N; k++)
            {
                frame[i][j][k] = new double[3];

            }
        }
    }
    double ****frame1 = new double*** [row];                //4-D Matrix for N frames of Video with 3 colors BGR
    for (int i = 0; i < row; i++)
    {
        frame1[i] = new double**[col];
        for(int j = 0; j< col; j++)
        {
            frame1[i][j] = new double*[N];
            for(int k =0; k <N; k++)
            {
                frame1[i][j][k] = new double[3];

            }
        }
    }
    Mat img;
    Mat imgbg;
    Mat imgbg1;
    capture.read(img);
    imgbg = img.clone();                // storing frame for further use
    imgbg1 = img.clone();               // storing frame for further use
    int k = 0;
    int r = 0;
    while(r<=398)			// here upper limit is defined as N-2 to include each and every frame.
    {
        capture.read(img);

        for (int i = 0; i < row; i++)                       //assigning values to 4-D Array frame from each frame
        {
            for(int j = 0; j< col; j++)
            {
                Vec3b intensity = img.at<Vec3b>(i, j);
                frame[i][j][k][0] = intensity.val[0];
                frame[i][j][k][1] = intensity.val[1];
                frame[i][j][k][2] = intensity.val[2];
                frame1[i][j][k][0] = intensity.val[0];
                frame1[i][j][k][1] = intensity.val[1];
                frame1[i][j][k][2] = intensity.val[2];
            }
        }
        k = k + 1;
        r++;
    }

    capture.release();
    double ****centroid = new double*** [row];          //4-D Array, which is storing mean value of clusters for K-Means
    for (int i = 0; i < row; i++)
    {
        centroid[i] = new double**[col];
        for(int j = 0; j< col; j++)
        {
            centroid[i][j] = new double*[K];
            for(int k =0; k <K; k++)
            {
                centroid[i][j][k] = new double[3];
                centroid[i][j][k][0] = 20*k + 12;
                centroid[i][j][k][1] = 4*k + 58;
                centroid[i][j][k][2] = 14*k + 76;
            }
        }
    }

    double ****A = new double***[row];              //4-D Array, which will store thei index of cluster from which that pixel belongs
    for (int i = 0; i < row; i++)
    {
        A[i] = new double**[col];
        for(int j = 0; j< col; j++)
        {
            A[i][j] = new double*[N];
            for(int k =0; k <N; k++)
            {
                A[i][j][k] = new double[3];
                for(int m = 0; m< 3; m++)
                {
                    A[i][j][k][m] = 0;
                }
            }
        }
    }

    double ****Kc = new double*** [row];            //4D Array, which will store the sum of the distances between pixel value and mean value, it's 4th column is storing the total number of points which belongs to that cluster
    for (int i = 0; i < row; i++)
    {
        Kc[i] = new double**[col];
        for (int j = 0; j < col; j++)
        {
            Kc[i][j] = new double*[K];
            for (int k = 0; k < K; k++)
            {
                Kc[i][j][k] = new double[6];
                for(int m = 0; m< 6; m++)
                {
                    if(m>=3)
                    {
                        Kc[i][j][k][m] = 1;
                    }
                    else
                    {
                        Kc[i][j][k][m] = 0;
                    }
                }
            }
        }
    }

    int u =0;
    while(u<5)                                                      // K-means iterations in while loop
    {
        u = u + 1;
        for(int i=0; i<row; i++)
        {
            for(int j=0; j<col; j++)
            {
                for(int k=0; k<N; k++)
                {
                    double min=48992349.00;
                    int centro=0;
                    for(int m=0; m<3; m++)
                    {
                        for(int l=0; l<K; l++)
                        {
                            double q = dist(centroid[i][j][l][m], frame[i][j][k][m]);
                            if(q<min)
                            {
                                min=q;
                                centro=l;
                            }
                        }
                        A[i][j][k][m] = centro;
                    }
                }
            }
        }
        for(int i=0; i<row; i++)
        {
            for(int j=0; j<col; j++)
            {
                for(int k=0; k<N; k++)
                {
                    for(int l=0; l<3; l++)
                    {
                        int p = A[i][j][k][l];
                        Kc[i][j][p][l+3] = Kc[i][j][p][l+3] + 1;
                        Kc[i][j][p][l] = Kc[i][j][p][l] + frame[i][j][k][l];
                    }
                }
            }
        }
        for(int i=0; i<row; i++)                                                //assigning the new mean values of clusters in 4D array Kc
        {
            for(int j=0; j<col; j++)
            {
                for(int k=0; k<K; k++)
                {
                    centroid[i][j][k][0] = Kc[i][j][k][0]/Kc[i][j][k][3];
                    centroid[i][j][k][1] = Kc[i][j][k][1]/Kc[i][j][k][4];
                    centroid[i][j][k][2] = Kc[i][j][k][2]/Kc[i][j][k][5];
                    Kc[i][j][k][0] = 0;
                    Kc[i][j][k][1] = 0;
                    Kc[i][j][k][2] = 0;
                    Kc[i][j][k][3] = 1;
                    Kc[i][j][k][4] = 1;
                    Kc[i][j][k][5] = 1;
                }
            }
        }
    }

    double ****var = new double ***[row];               //4-D array to store variance
    for (int i = 0; i < row; i++)
    {
        var[i] = new double **[col];
        for(int j = 0; j< col; j++)
        {
            var[i][j] = new double*[K];
            for(int k =0; k <K; k++)
            {
                var[i][j][k] = new double[6];
                var[i][j][k][0] = 0.00001;
                var[i][j][k][1] = 0.00001;
                var[i][j][k][2] = 0.00001;
                var[i][j][k][3] = 1;
                var[i][j][k][4] = 1;
                var[i][j][k][5] = 1;
            }
        }
    }

    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            for(int m=0; m<3; m++)
            {
                for(int k=0; k<N; k++)
                {
                    int p = A[i][j][k][m];
                    double a[3];
                    a[m] = centroid[i][j][p][m];
                    var[i][j][p][m]= var[i][j][p][m]+(a[m]-frame[i][j][k][m])*(a[m]-frame[i][j][k][m]);
                    var[i][j][p][m+3]= var[i][j][p][m+3]+1;
                }
            }
        }
    }

    for(int i=0; i<row; i++)                     // storing the variances,
    {
        for(int j=0; j<col; j++)
        {
            for(int k=0; k<K; k++)
            {
                var[i][j][k][0]= (var[i][j][k][0])/var[i][j][k][3];
                var[i][j][k][1]= (var[i][j][k][1])/var[i][j][k][4];
                var[i][j][k][2]= var[i][j][k][2]/var[i][j][k][5];
                var[i][j][k][3]= var[i][j][k][3]/N;                             //here we are storing the pi value of each gaussian of blue
                var[i][j][k][4]= var[i][j][k][4]/N;                             //here we are storing the pi value of each gaussian of green
                var[i][j][k][5]= var[i][j][k][5]/N;                             //here we are storing the pi value of each gaussian of red
            }
        }
    }


    double learning = 0.01;                         //defining learning rate

    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            for(int k=0; k<N; k++)
            {
                double t = frame[i][j][k][0];
                double t1 = frame[i][j][k][1];
                double t2 = frame[i][j][k][2];
                vector<int> vec;
                vector<int> vec1;
                vector<int> vec2;
                for(int l=0; l<K; l++)
                {
                    double m = centroid[i][j][l][0];
                    double v = sqrt(var[i][j][l][0]);
                    double m1 = centroid[i][j][l][1];
                    double v1 = sqrt(var[i][j][l][1]);
                    double m2 = centroid[i][j][l][2];
                    double v2 = sqrt(var[i][j][l][2]);                              //parameters without numerics are for blue shades.
                    if(t <= m + 2.5*v && t>= m-2.5*v)                               //if new point belongs to any 1 of the gaussian then update the following parameters
                    {
                        double ro = learning*Gauss(t, m, v);
                        centroid[i][j][l][0] = (1-ro)*m + ro*t;
                        var[i][j][l][0] = (1-ro)*v*v + ro*(t-m)*(t-m);
                        var[i][j][l][3] = (1-learning)*(var[i][j][l][3]) + learning;
                    }
                    else                                                    // if that point does not match that gaussian, then add that gaussian index to vector vec
                    {
                        vec.push_back(l);
                    }
                    if(t1 <= m1 + 2.5*v1 && t1>= m1-2.5*v1)                     //same above steps repeating for Green shades
                    {
                        double ro1 = learning*Gauss(t1, m1, v1);
                        centroid[i][j][l][1] = (1-ro1)*m1 + ro1*t1;
                        var[i][j][l][1] = (1-ro1)*v1*v1 + ro1*(t1-m1)*(t1-m1);
                        var[i][j][l][4] = (1-learning)*(var[i][j][l][4]) + learning;
                    }
                    else
                    {
                        vec1.push_back(l);
                    }
                    if(t2 <= m2 + 2.5*v2 && t2>= m2-2.5*v2)                 //same above steps repeating for Red shades
                    {
                        double ro2 = learning*Gauss(t2, m2, v2);
                        centroid[i][j][l][2] = (1-ro2)*m2 + ro2*t2;
                        var[i][j][l][2] = (1-ro2)*v2*v2 + ro2*(t2-m2)*(t2-m2);
                        var[i][j][l][5] = (1-learning)*(var[i][j][l][5]) + learning;
                    }
                    else
                    {
                        vec2.push_back(l);
                    }
                }
                if(vec.size() == K)                 // here we are checking if the point did not match any of the gaussian then we will replace the least probable gaussian with the new one, designed by us.
                {
                    double min = 4562741.00;
                    int o = 445;
                    for(int l=0; l<K; l++)
                    {
                        double z1 = var[i][j][l][0];
                        if(z1 <= 0)
                        {
                            z1 = 0.1;
                        }
                        double s = var[i][j][l][3]/sqrt(z1);
                        if(s < min)
                        {
                            min = s;
                            o = l;
                        }
                    }
                    centroid[i][j][o][0] = t;
                    var[i][j][o][0] = 35;
                    var[i][j][o][3] = 0.00025;
                }
                else                                //if atleast single match occur but not all matched then update the probablity value of non matched gaussians
                {
                    int a1 = vec.size();
                    while(a1 > 0)
                    {
                        int a2 = vec[a1-1];
                        var[i][j][a2][3] = (1-learning)*(var[i][j][a2][3]);
                        a1 = a1 - 1;
                    }
                }
                if(vec1.size() == K)                    //same above steps repeating for Green shades
                {
                    double min = 4562741.00;
                    int o = 445;
                    for(int l=0; l<K; l++)
                    {
                        double z1 = var[i][j][l][1];
                        if(z1 <= 0)
                        {
                            z1 = 0.1;
                        }
                        double s = var[i][j][l][4]/sqrt(z1);
                        if(s < min)
                        {
                            min = s;
                            o = l;
                        }
                    }
                    centroid[i][j][o][1] = t1;
                    var[i][j][o][1] = 35;
                    var[i][j][o][4] = 0.00025;
                }
                else
                {
                    int a1 = vec1.size();
                    while(a1 > 0)
                    {
                        int a2 = vec1[a1-1];
                        var[i][j][a2][4] = (1-learning)*(var[i][j][a2][4]);
                        a1 = a1 - 1;
                    }
                }
                if(vec2.size() == K)                                        //same above steps repeating for Red shades
                {
                    double min = 4562741.00;
                    int o = 445;
                    for(int l=0; l<K; l++)
                    {
                        double z2 = var[i][j][l][2];
                        if(z2 <= 0)
                        {
                            z2 = 0.1;
                        }
                        double s = var[i][j][l][5]/sqrt(z2);
                        if(s < min)
                        {
                            min = s;
                            o = l;
                        }
                    }
                    centroid[i][j][o][2] = t2;
                    var[i][j][o][2] = 35;
                    var[i][j][o][5] = 0.00025;
                }
                else
                {
                    int a1 = vec2.size();
                    while(a1 > 0)
                    {
                        int a2 = vec2[a1-1];
                        var[i][j][a2][5] = (1-learning)*(var[i][j][a2][5]);
                        a1 = a1 - 1;
                    }
                }
            }
        }
    }

                        //Now here we are ready with 3 gaussians for each point and now we will segment out background and foreground.
	
	//VideoWriter out_capture1("Background.avi", 0, 25, Size(352,240), 1);
	
    for(int n=0; n<N; n++)
    {
        for(int i=0; i<row; i++)
        {
            for(int j=0; j<col; j++)
            {
                double max = 0;
                int o = 2;
                double max1 = 0;
                int o1 = 2;
                double max2 = 0;
                int o2 = 2;
                for(int l=0; l<K; l++)                  // finding out maximum probable gaussian for each shades (maximum probable means responsible for background formation)
                {
                    double s = var[i][j][l][3]/sqrt(var[i][j][l][0]);
                    double s1 = var[i][j][l][4]/sqrt(var[i][j][l][1]);
                    double s2 = var[i][j][l][5]/sqrt(var[i][j][l][2]);
                    if(s > max)
                    {
                        max = s;
                        o = l;
                    }
                    if(s1 > max1)
                    {
                        max1 = s1;
                        o1 = l;
                    }
                    if(s2 > max2)
                    {
                        max2 = s2;
                        o2 = l;
                    }
                }
                double t = frame[i][j][n][0];
                double m = centroid[i][j][o][0];
                double v = sqrt(var[i][j][o][0]);
                double t1 = frame[i][j][n][1];
                double m1 = centroid[i][j][o1][1];
                double v1 = sqrt(var[i][j][o1][1]);
                double t2 = frame[i][j][n][2];
                double m2 = centroid[i][j][o2][2];
                double v2 = sqrt(var[i][j][o2][2]);

          
                if(t > m + 2.5*v || t< m-2.5*v)                 //Now we are iterating from frame 1 to N and if the point belongs to most probable gaussian then we will not change that pixel value. otherwise we will replace the pixel value with the mean of most probable gaussian
                {
                    frame[i][j][n][0] = centroid[i][j][o][0];
                }
                if(t1 > m1 + 2.5*v1 || t1< m1-2.5*v1)            //same above steps repeating for Green shades
                {
                    frame[i][j][n][1] = centroid[i][j][o1][1];
                }
                if(t2 > m2 + 2.5*v2 || t2< m2-2.5*v2)               //same above steps repeating for Red shades
                {
                    frame[i][j][n][2] = centroid[i][j][o2][2];
                }

                imgbg.at<Vec3b>(i, j).val[0] = frame[i][j][n][0];       // once we are finished we are collecting nth frames pixels and generating our background frames
                imgbg.at<Vec3b>(i, j).val[1] = frame[i][j][n][1];
                imgbg.at<Vec3b>(i, j).val[2] = frame[i][j][n][2];
            }
        }
				imshow("background",imgbg);
				waitKey(25);
				//out_capture1<<imgbg;
    }											// here we have recieved our background video.
										
     
    //VideoWriter out_capture("Foreground.avi", 0, 25, Size(352,240), 1);                                   
    
    for(int n=0; n<N; n++)
    {
        for(int i=0; i<row; i++)
        {
            for(int j=0; j<col; j++)            // this process is same as above and to generate foreground video file
            {
                double max = 0;
                int o = 2;
                double max1 = 0;
                int o1 = 2;
                double max2 = 0;
                int o2 = 2;
                for(int l=0; l<K; l++)
                {
                    double s = var[i][j][l][3]/sqrt(var[i][j][l][0]);    // finding out maximum probable gaussian for each shades (maximum probable means responsible for background formation)
                    double s1 = var[i][j][l][4]/sqrt(var[i][j][l][1]);
                    double s2 = var[i][j][l][5]/sqrt(var[i][j][l][2]);

                    if(s > max)
                    {
                        max = s;
                        o = l;
                    }
                    if(s1 > max1)
                    {
                        max1 = s1;
                        o1 = l;
                    }
                    if(s2 > max2)
                    {
                        max2 = s2;
                        o2 = l;
                    }
                }
                double t = frame1[i][j][n][0];
                double m = centroid[i][j][o][0];
                double v = sqrt(var[i][j][o][0]);
                double t1 = frame1[i][j][n][1];
                double m1 = centroid[i][j][o1][1];
                double v1 = sqrt(var[i][j][o1][1]);
                double t2 = frame1[i][j][n][2];
                double m2 = centroid[i][j][o2][2];
                double v2 = sqrt(var[i][j][o2][2]);
                if(t <= m + 2.5*v && t>= m-2.5*v)           //Now we are iterating from frame 1 to N and if the point belongs to most probable gaussian then we will set that pixel to white color (because thats the background part), otherwise we will leave the pixel unchanged
                {
                    frame1[i][j][n][0] = 255;
                }
                if(t1 <= m1 + 2.5*v1 && t1>= m1-2.5*v1)         //same above steps repeating for Green shades
                {
                    frame1[i][j][n][1] = 255;
                }
                if(t2 <= m2 + 2.5*v2 && t2>= m2-2.5*v2)             //same above steps repeating for Red shades
                {
                    frame1[i][j][n][2] = 255;
                }
                imgbg1.at<Vec3b>(i, j).val[0] = frame1[i][j][n][0];      // once we are finished we are collecting nth frames pixels and generating our foreground frames
                imgbg1.at<Vec3b>(i, j).val[1] = frame1[i][j][n][1];
                imgbg1.at<Vec3b>(i, j).val[2] = frame1[i][j][n][2];
				
			}
		}
				imshow("Foreground",imgbg1);
				waitKey(25);
				//out_capture<<imgbg1;
	}	
                                                                                // here we have recieved our foreground video.
}