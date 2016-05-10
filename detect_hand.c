//
//  hand.c
//  Opencv
//
//  Created by Jeevjyot Singh Chhabda on 5/2/16.
//  Copyright © 2016 Jeevjyot Singh Chhabda. All rights reserved.
//
/* Algorithm works in the following way
 
 1. Capture the video frame.
 2. Start Recording .
 3. Initialise the window
 4. get the camera feed from webcam.
 5. Filer and find the thresold image.
 6. Find countour.
 7. Find_Convex_Hull.
 8. Find_fingers, count the number of fingers.
 
 */

#include <stdio.h>
#include <math.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

#define VIDEO_FILE	"video.avi"  //video file name
#define VIDEO_FORMAT	CV_FOURCC('M', 'J', 'P', 'G') //four character code, this determines the code for v
#define NUM_FINGERS	5       // number of finger, wont work for six fingers.
#define NUM_DEFECTS	8
//Well, this is used to draw the lines on the hand, using RGB all colors can be formed
#define RED     CV_RGB(255, 0, 0)
#define GREEN   CV_RGB(0, 255, 0)
#define BLUE    CV_RGB(0, 0, 255)
#define YELLOW  CV_RGB(255, 255, 0)
#define PURPLE  CV_RGB(255, 0, 255)
#define GREY    CV_RGB(200, 200, 200)

struct ctx {
    
    CvMemStorage	*hull_st;
    CvMemStorage	*contour_st;
    CvMemStorage	*temp_st;
    CvMemStorage	*defects_st;
    
    IplConvKernel	*kernel;	/* Kernel for morph operations */
    
    int		num_fingers;
    int		hand_radius;
    int		num_defects;
    
    CvCapture	*capture;	/* Capture handle */
    CvVideoWriter	*writer;	/* File recording handle */
    
    IplImage	*image;		/* Input image */
    IplImage	*thr_image;	/* After filtering and thresholding */
    IplImage	*temp_image1;	/* Temporary image (1 channel) */
    IplImage	*temp_image3;	/* Temporary image (3 channels) */
    
    CvSeq		*contour;	/* Hand contour */
    CvSeq		*hull;		/* Hand convex hull */
    
    CvPoint		hand_center;
    CvPoint		*fingers;	/* Detected fingers positions */
    CvPoint		*defects;	/* Convexity defects depth points */
    
    
};
/*
 this initialise the webcame, device number is 0.
 */
void init_capture(struct ctx *ctx)
{
    //cvCapture is used to initialise the camera and provide the feed.
    ctx->capture = cvCaptureFromCAM(0);
    if (!ctx->capture) {
        fprintf(stderr, "Error initializing capture\n");
        exit(1);
    }
    //cvQueryFrame() returns the pointer to OpenCV's "private" internal buffer which it always fills with  last grabbed frame.
    ctx->image = cvQueryFrame(ctx->capture);
}

void init_recording(struct ctx *ctx)
{
    int fps, width, height;
    //fps is frame per second.
    //returns the frame per second.
    fps = cvGetCaptureProperty(ctx->capture, CV_CAP_PROP_FPS);
    //returns the height
    printf("Frame per second =%d",fps);
    width = cvGetCaptureProperty(ctx->capture, CV_CAP_PROP_FRAME_WIDTH);
    //returns the frame width
    printf("Width of the video is=%d",width);
    //returns the height
    height = cvGetCaptureProperty(ctx->capture, CV_CAP_PROP_FRAME_HEIGHT);
    printf("Height of the video is=%d",height);
    if (fps < 0)
        fps = 10;
    
    //cvCreateVideoWriter Constructor writes the video file
    ctx->writer = cvCreateVideoWriter(VIDEO_FILE, VIDEO_FORMAT, fps,
                                      cvSize(width, height), 1);
    
    if (!ctx->writer) {
        fprintf(stderr, "Error initializing video writer\n");
        exit(1);
    }
}

void init_windows(void)
{
    cvNamedWindow("output", CV_WINDOW_AUTOSIZE); // basic camera feed
    cvNamedWindow("thresholded", CV_WINDOW_AUTOSIZE);   // this show the thesholded value, removes the background.
    //moves window the speicified position
    cvMoveWindow("output", 50, 50);
    //keep the both windows side by side
    cvMoveWindow("thresholded", 700, 50);
}

void init_ctx(struct ctx *ctx)
{
    //Creating image of size(width),depth is 8, bit value, 1 channel image
    ctx->thr_image = cvCreateImage(cvGetSize(ctx->image), 8, 1);//thresholded image.
    ctx->temp_image1 = cvCreateImage(cvGetSize(ctx->image), 8, 1);  //1 channel image
    ctx->temp_image3 = cvCreateImage(cvGetSize(ctx->image), 8, 3);  //3 channel image
    ctx->kernel = cvCreateStructuringElementEx(9, 9, 4, 4, CV_SHAPE_RECT,
                                               NULL);
    ctx->contour_st = cvCreateMemStorage(0);
    ctx->hull_st = cvCreateMemStorage(0);
    ctx->temp_st = cvCreateMemStorage(0);
    ctx->fingers = calloc(NUM_FINGERS + 1, sizeof(CvPoint)); //initialise the memory; number of fingers
    ctx->defects = calloc(NUM_DEFECTS, sizeof(CvPoint));//depth point
}

void filter_and_threshold(struct ctx *ctx)
{
    
    /* Soften image */
    //source image is the captured image, detined image we have to create
    //smoothing is also called as blurring,
    cvSmooth(ctx->image, ctx->temp_image3, CV_GAUSSIAN, 11, 11, 0, 0);
    /* Remove some impulsive noise */
    cvSmooth(ctx->temp_image3, ctx->temp_image3, CV_MEDIAN, 11, 11, 0, 0);
    
    cvCvtColor(ctx->temp_image3, ctx->temp_image3, CV_BGR2HSV);
    //cvNamedWindow("G",CV_WINDOW_AUTOSIZE);
   // cvShowImage("G", ctx->temp_image3);
    /*
     * Apply threshold on HSV values to detect skin color
     */
    cvInRangeS(ctx->temp_image3,
               cvScalar(0, 55, 90, 255),
               cvScalar(28, 175, 230, 255),
               ctx->thr_image);
    
    /* Apply morphological opening */
    cvMorphologyEx(ctx->thr_image, ctx->thr_image, NULL, ctx->kernel,
                   CV_MOP_OPEN, 1);
    cvSmooth(ctx->thr_image, ctx->thr_image, CV_GAUSSIAN, 3, 3, 0, 0);
}

void find_contour(struct ctx *ctx)
{
    double area, max_area = 0.0;
    CvSeq *contours, *tmp, *contour = NULL;
    
    /* cvFindContours modifies input image, so make a copy */
    cvCopy(ctx->thr_image, ctx->temp_image1, NULL);
    cvFindContours(ctx->temp_image1, ctx->temp_st, &contours,sizeof(CvContour), CV_RETR_EXTERNAL,
    CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
    
    /* Select contour having greatest area */
    for (tmp = contours; tmp; tmp = tmp->h_next) {
        area = fabs(cvContourArea(tmp, CV_WHOLE_SEQ, 0));
        if (area > max_area) {
            max_area = area;
            contour = tmp;
        }
    }
    
    /* Approximate contour with poly-line */
    if (contour) {
        contour = cvApproxPoly(contour, sizeof(CvContour),
                               ctx->contour_st, CV_POLY_APPROX_DP, 2,
                               1);
        ctx->contour = contour;
    }
}

void find_convex_hull(struct ctx *ctx)
{
    CvSeq *defects;
    CvConvexityDefect *defect_array;
    int i;
    int x = 0, y = 0;
    int dist = 0;
    
    ctx->hull = NULL;
    
    if (!ctx->contour)
        return;
    
    ctx->hull = cvConvexHull2(ctx->contour, ctx->hull_st, CV_CLOCKWISE, 0);
    
    if (ctx->hull) {
        
        /* Get convexity defects of contour w.r.t. the convex hull */
        defects = cvConvexityDefects(ctx->contour, ctx->hull,
                                     ctx->defects_st);
        
        if (defects && defects->total) {
            defect_array = calloc(defects->total,
                                  sizeof(CvConvexityDefect));
            cvCvtSeqToArray(defects, defect_array, CV_WHOLE_SEQ);
            
            /* Average depth points to get hand center */
            for (i = 0; i < defects->total && i < NUM_DEFECTS; i++) {
                x += defect_array[i].depth_point->x;
                y += defect_array[i].depth_point->y;
                
                ctx->defects[i] = cvPoint(defect_array[i].depth_point->x,
                                          defect_array[i].depth_point->y);
            }
            
            x /= defects->total;
            y /= defects->total;
            
            ctx->num_defects = defects->total;
            ctx->hand_center = cvPoint(x, y);
            
            /* Compute hand radius as mean of distances of
             defects' depth point to hand center */
            for (i = 0; i < defects->total; i++) {
                int d = (x - defect_array[i].depth_point->x) *
                (x - defect_array[i].depth_point->x) +
                (y - defect_array[i].depth_point->y) *
                (y - defect_array[i].depth_point->y);
                
                dist += sqrt(d);
            }
            
            ctx->hand_radius = dist / defects->total;
            free(defect_array);
        }
    }
}

void find_fingers(struct ctx *ctx)
{
    int n;
    int i;
    CvPoint *points;
    CvPoint max_point;
    int dist1 = 0, dist2 = 0;
    
    ctx->num_fingers = 0;
    
    if (!ctx->contour || !ctx->hull)
        return;
    
    n = ctx->contour->total;
    points = calloc(n, sizeof(CvPoint));
    
    cvCvtSeqToArray(ctx->contour, points, CV_WHOLE_SEQ);
    
    /*
     * Fingers are detected as points where the distance to the center
     * is a local maximum
     */
   // printf("Total number of fingers %d",n);
    for (i = 0; i < n; i++) {
        int dist;
        int cx = ctx->hand_center.x;
        int cy = ctx->hand_center.y;
        
        dist = (cx - points[i].x) * (cx - points[i].x) +
        (cy - points[i].y) * (cy - points[i].y);
        
        if (dist < dist1 && dist1 > dist2 && max_point.x != 0
            && max_point.y < cvGetSize(ctx->image).height - 10) {
            
            ctx->fingers[ctx->num_fingers++] = max_point;
            if (ctx->num_fingers >= NUM_FINGERS + 1)
                break;
        }
        
        dist2 = dist1;
        dist1 = dist;
        max_point = points[i];
    }
    
    free(points);
}

void display(struct ctx *ctx)
{
    int i;
    
    if (ctx->num_fingers == NUM_FINGERS) {
        
#if defined(SHOW_HAND_CONTOUR)
        cvDrawContours(ctx->image, ctx->contour, BLUE, GREEN, 0, 1,
                       CV_AA, cvPoint(0, 0));
#endif
        cvCircle(ctx->image, ctx->hand_center, 5, PURPLE, 1, CV_AA, 0);
        cvCircle(ctx->image, ctx->hand_center, ctx->hand_radius,
                 RED, 1, CV_AA, 0);
        
        for (i = 0; i < ctx->num_fingers; i++) {
            
            cvCircle(ctx->image, ctx->fingers[i], 10,
                     GREEN, 3, CV_AA, 0);
            
            cvLine(ctx->image, ctx->hand_center, ctx->fingers[i],
                   YELLOW, 1, CV_AA, 0);
        }
        
        for (i = 0; i < ctx->num_defects; i++) {
            cvCircle(ctx->image, ctx->defects[i], 2,
                     GREY, 2, CV_AA, 0);
        }
    }
    
    cvShowImage("output", ctx->image);
    cvShowImage("thresholded", ctx->thr_image);
}

int main(int argc, char **argv)
{
    struct ctx ctx = { };
    int key;
    
    init_capture(&ctx);     // Initialse the video, starts the webcame
    init_recording(&ctx);   //start the recording, initialise the window, capture the image from video, starts writing video file.
    init_windows(); // initialise the window.
    init_ctx(&ctx);
    //continue capturing the camera feed, till the time user doesnt hit q
    do {
        ctx.image = cvQueryFrame(ctx.capture);
        //supply image to the following function
        filter_and_threshold(&ctx);
        find_contour(&ctx);
        find_convex_hull(&ctx);
        find_fingers(&ctx);
        
        display(&ctx);
        cvWriteFrame(ctx.writer, ctx.image);
        
        key = cvWaitKey(1);
    } while (key != 'q');       //continue capturing the camera feed, till the time user doesnt hit q
    
    return 0;
}
