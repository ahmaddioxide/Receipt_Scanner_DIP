import cv2
import numpy as np
import utlis

# Set up configuration
webCamFeed = False  # Set to True if using webcam feed, False if using an image file
pathImage = "5.jpg"  # Path to the image file if not using webcam feed
cap = cv2.VideoCapture(0)  # Webcam index, change to the appropriate index if needed
cap.set(10, 160)  # Set the webcam brightness (adjust as needed)
heightImg = 700  # Desired height of the image
widthImg = 500  # Desired width of the image

# Initialize trackbars
utlis.initializeTrackbars()

count = 0  # Counter for saving images

while True:
    # Read the input image
    if webCamFeed:
        success, img = cap.read()  # Capture frame from the webcam
    else:
        img = cv2.imread(pathImage)  # Read image from the specified file

    img = cv2.resize(img, (widthImg, heightImg))  # Resize the input image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Create a blank image for testing/debugging if required
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Apply Gaussian blur to the grayscale image
    thres = utlis.valTrackbars()  # Get trackbar values for thresholds
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # Apply Canny edge detection
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # Apply dilation
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # Apply erosion

    # Find all contours
    imgContours = img.copy()  # Copy image for display purposes
    imgBigContour = img.copy()  # Copy image for display purposes
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find all contours
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # Draw all detected contours

    # Find the biggest contour
    biggest, maxArea = utlis.biggestContour(contours)  # Find the biggest contour
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # Draw the biggest contour
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)  # Draw a rectangle around the biggest contour
        pts1 = np.float32(biggest)  # Prepare points for warp
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Prepare points for warp
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Calculate the perspective transform matrix
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # Apply perspective transform

        # Remove 20 pixels from each side
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))  # Resize the warped image

        # Apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # Convert the warped image to grayscale
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)  # Apply adaptive thresholding
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)  # Invert the thresholded image
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)  # Apply median blur to the thresholded image

        # Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # Labels for Display
    labels = [["Orig", "GreyScaled", "After Thresholding", "Contours"],
              ["Biggest-Contour", "Warp Perspective", "Warp Grey", "Adaptive Thresholding"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, labels)  # Stack images for display
    cv2.imshow("Result", stackedImage)  # Display the stacked image

    # Save image when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("updatedSlip" + str(count) + ".jpg", imgWarpColored)  # Save the warped image
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)  # Display the "Scan Saved" message
        cv2.waitKey(300)
        count += 1  # Increment the counter
