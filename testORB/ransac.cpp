int surfMatch::estimatWithRansacBase(
        const std::vector<cv::KeyPoint>& keyPoint1,
        const std::vector<cv::KeyPoint>& keyPoint2,
        const std::vector<cv::DMatch>& matchePoints,
        std::vector<cv::DMatch>& goodMatchePoints,
        cv::Mat_<double>& H,
        double inlier_thresh)
{
    std::vector<int>                tempIndex;
    size_t                          tempNum;
    int                             maxLoopCount = 1500;
    int                             loopCount = 0;
    cv::Mat_<double>                  htemp;
    std::vector<cv::Point_<double>>   pt1(4);
    std::vector<cv::Point_<double>>   pt2(4);

    size_t corspNum = matchePoints.size();

    if (corspNum <= 4) return -1;

    tempIndex.resize((corspNum / 100 + 1) * 100);
    goodMatchePoints.clear();

    while (loopCount < maxLoopCount)
    {
        int     sample[4];
        for (int i = 0; i < 4; i++)
        {
            static int s = 0;
            int        j;

            if (s++ == 0) srand(0);

            int pos = (int)((double)corspNum * rand() / (RAND_MAX + 1.0F));
            for (j = 0; j < i; j++)
            {
                if (pos == sample[j]) break;
            }
            if (j < i)
            {
                i--;
                continue;
            }
            else
            {
                sample[i] = pos;
            }
            pt1[i].x = keyPoint1[matchePoints[pos].queryIdx].pt.x;
            pt1[i].y = keyPoint1[matchePoints[pos].queryIdx].pt.y;
            pt2[i].x = keyPoint2[matchePoints[pos].trainIdx].pt.x;
            pt2[i].y = keyPoint2[matchePoints[pos].trainIdx].pt.y;
        }

        //if ((!IsGoodSample(pt1)) || (!IsGoodSample(pt2))) { loopCount++; continue; }

        if (ComputeAffine(pt1, pt2, htemp) < 0) { loopCount++; continue; }
        double *htempptr = htemp.ptr<double>(0);
        tempNum = 0;
        for (size_t i = 0; i < corspNum; i++)
        {
            if (i == 18)
                i = i;
            double xx2, yy2, ww2, err;
            xx2 = htempptr[0] * keyPoint2[matchePoints[i].trainIdx].pt.x + htempptr[1] * keyPoint2[matchePoints[i].trainIdx].pt.y + htempptr[2];
            yy2 = htempptr[3] * keyPoint2[matchePoints[i].trainIdx].pt.x + htempptr[4] * keyPoint2[matchePoints[i].trainIdx].pt.y + htempptr[5];
            ww2 = htempptr[6] * keyPoint2[matchePoints[i].trainIdx].pt.x + htempptr[7] * keyPoint2[matchePoints[i].trainIdx].pt.y + htempptr[8];
            xx2 /= ww2;
            yy2 /= ww2;
            err = (xx2 - keyPoint1[matchePoints[i].queryIdx].pt.x)*(xx2 - keyPoint1[matchePoints[i].queryIdx].pt.x) +
                  (yy2 - keyPoint1[matchePoints[i].queryIdx].pt.y)*(yy2 - keyPoint1[matchePoints[i].queryIdx].pt.y);
            if (err < inlier_thresh)
            {
                tempIndex[tempNum] = i;
                tempNum++;
            }
            else
            {
                err = err;
            }
        }

        if (tempNum > goodMatchePoints.size())
        {
            H = htemp.clone();

            goodMatchePoints.resize(tempNum);
            for (size_t i = 0; i < tempNum; i++)
            {
                goodMatchePoints[i] = matchePoints[tempIndex[i]];
            }

            static const double p = log(1.0F - 0.99F);
            double e = static_cast<double>(tempNum) / static_cast<double>(corspNum);
            double e_thresh = pow((1 - pow(10, -2.0F / maxLoopCount)), 0.25);
            if (e > e_thresh)
            {
                double N = (p / log(1 - e*e*e*e));
                maxLoopCount = static_cast<int>(N);
            }
        }
        loopCount++;
    }

    pt1.resize(goodMatchePoints.size());
    pt2.resize(goodMatchePoints.size());
    for (int i = 0; i < goodMatchePoints.size(); i++)
    {
        pt1[i].x = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
        pt1[i].y = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
        pt2[i].x = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
        pt2[i].y = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
    }
    if (ComputeAffine(pt1, pt2, H) < 0)
    {
        return -2;
    }

    return 0;
}