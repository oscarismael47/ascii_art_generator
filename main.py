import cv2
import numpy as np

def draw_char(image_base,image_char,image_rgb):
    ## para green colormap
    #image_rgb = cv2.applyColorMap(image_rgb, cv2.COLORMAP_DEEPGREEN)
    h,w = image_char.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    color = (255, 255, 255)
    thickness = 1
    cell_width =  10

    for i in range(h): #
        for j in range(w): #
            color = (int(image_rgb[i,j][0]),int(image_rgb[i,j][1]),int(image_rgb[i,j][2]))
            image_base = cv2.putText(image_base,image_char[i,j], (j*cell_width,i*cell_width), font, fontScale,color, thickness, cv2.LINE_AA)

    return image_base




density_char = ['n','@','W','$','9','8','7','6','5','4','3','2','1','0','¡','a','b','c',':','=','.',' ',' ',' ',' ',' ',' ',' ',' ']
density_char.reverse()

cap = cv2.VideoCapture("video3.mp4")
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_width, frame_height = 1100,700
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output.mp4', 
                         fourcc,
                         20, size)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame_ori = cap.read()
    
    frame_w, frame_h = 100,100
    frame_base = np.zeros((frame_h*10,frame_w*10,3),dtype=np.uint8)

    if ret == True:
        frame = cv2.resize(frame_ori.copy(),(frame_w,frame_h))	
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_max = np.max(frame_gray)
        frame_gray_flatten = frame_gray.flatten()
        factor = (len(density_char)-1)/frame_gray_max
     

        frame_char = np.array([density_char[int(pixel*factor)] for pixel in frame_gray_flatten])
        frame_char = frame_char.reshape((frame_h, frame_w))
        frame_base[:,:,:] = 0
        frame_base = draw_char(frame_base,frame_char,frame)


        frame_ori= cv2.resize(frame_ori,(900,700))
        frame_base= cv2.resize(frame_base,(900,700))

        frame_result = np.hstack((frame_ori,frame_base))
        frame_result= cv2.resize(frame_result,(1100,700))
        # Display the resulting frame
        

        output.write(frame_result)

        cv2.imshow('Frame char',frame_result)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else: 
        break
# When everything done, release the video capture object
cap.release()
output.release()
# Closes all the frames
cv2.destroyAllWindows()