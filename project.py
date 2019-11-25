
def read_cam():
    #cap = cv2.VideoCapture('Cars moving on road Stock Footage - Free Download.mp4')
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('Busy traffic in Kolkata - West Bengal.mp4')
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_count = frame_count+1
        if ret == True:
            #cv2.imwrite("test.jpg",frame)
            #img_path = "test.jpg"
            boxes, scores, classes, nums = yolo(decode_image(frame))
            img= draw_outputs(frame, (boxes, scores, classes, nums), class_names)
            cv2.imshow("output",img)
            #cv2.imwrite('C://Users/DEBANJAN GHOSH/Downloads/video/image'+str(frame_count)+'.jpeg',img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

'''
def decode_image(img_path):
    #img = cv2.imread(img_path)
    img = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img_test = transform_images(img, 416)
    return (img_test)
'''
def decode_image(img):
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, 0)
    img_test = transform_images(img, 416)
    return (img_test)

def calculate(outputs, class_names):
    vehicle = 0
    pedestrian = 0
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    #print(classes)

    for i in range(nums):
        if (class_names[int(classes.numpy()[i])]) == "person":
            pedestrian = pedestrian+1
        elif (class_names[int(classes.numpy()[i])]) == "car" or (class_names[int(classes.numpy()[i])]) == "truck" or (class_names[int(classes.numpy()[i])]) == "bus" or (class_names[int(classes.numpy()[i])]) == 'motorbike':
            vehicle = vehicle+1
    return(vehicle,pedestrian)
