import sys, os.path
import cv2

if len(sys.argv) < 2:
    print ("Usage: pyton3 cut_images.py directory_of_images")
    sys.exit()

print (sys.argv[1])
path = sys.argv[1]
path_labels = path+"/labels.csv"
if not os.path.exists(path_labels):
    print ("Required file: " + path_labels + " is not found.")
    sys.exit()

print (path)
print ("crowdai" in path)

is_crowdai = "crowdai" in path

id_img = 0

for line in open(path_labels):

    items = line.split(',') if is_crowdai else line.split()
    print(items)

    if is_crowdai:
        if not items[5] == str('Car'):
            continue
    else:
        if not items[6] == str('"car"'):
            continue

    fname = items[4] if is_crowdai else items[0]
    img = cv2.imread(path+'/'+fname)

    try:
        if is_crowdai:
            x1,y1 = int(items[0]), int(items[1])
            x2,y2 = int(items[2]), int(items[3])
        else:
            x1,y1 = int(items[1]), int(items[2])
            x2,y2 = int(items[3]), int(items[4])
    except:
        continue

    print(x1,y1,x2,y2)
    img_cropped = img[y1:y2,x1:x2]

    # cv2.rectangle(img,(x1,y1),(x2,y2),(200,200,0),5)
    # cv2.imshow("img",img)
    # cv2.imshow("cropped",img_cropped)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    path_write = "../imgs_car/"+"%05d"%id_img+'-'+ fname
    id_img += 1
    print(path_write)
    cv2.imwrite(path_write, img_cropped)
