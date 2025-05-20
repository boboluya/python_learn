import cv2


def resizeimg(path:str,box:list,target_size:list)->list:
    img = cv2.imread(path)
    org_shar = img.shape
    orignx = org_shar[1]
    origny = org_shar[0]
    # scalex = target_size[0] / orignx
    # scaley = target_size[1] / origny

    # x = int(box[0] * scaley)
    # y = int(box[1] * scalex)
    # w = int(box[2] * scaley)
    # h = int(box[3] * scalex)
    x = box[0]/orignx
    w = box[2]/orignx
    y=box[1]/origny
    h=box[3]/origny

    # img = cv2.resize(img, (target_size[0] ,target_size[1]))
    # cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return [x,y,w,h]

def revertimg(path:str,box:list,target_size:list):
    img = cv2.imread(path)
    org_shar = img.shape
    orignx = org_shar[1]
    origny = org_shar[0]
    # scalex = target_size[0] / orignx
    # scaley = target_size[1] / origny
    # print(scalex,scaley)

    # x = int(box[0] / scaley)
    # y = int(box[1] / scalex)
    # w = int(box[2] / scaley)
    # h = int(box[3] / scalex)
    
    x = int(box[0]*orignx)
    w = int(box[2]*orignx)
    y = int(box[1]*origny)
    h = int(box[3]*origny)

    # img = cv2.resize(img, (target_size[0] ,target_size[1]))
    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
    return img

if __name__ == "__main__":
    res = resizeimg("C:\\Users\\bobol\\Desktop\\CNN\\faces\\train\\0--Parade/0_Parade_marchingband_1_849.jpg",[449, 330, 122, 149],[128,128])
    #    res = [i+100 for i in res]
    img = revertimg("C:\\Users\\bobol\\Desktop\\CNN\\faces\\train\\0--Parade/0_Parade_marchingband_1_849.jpg",res,[128,128])
    cv2.imshow("sss",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
