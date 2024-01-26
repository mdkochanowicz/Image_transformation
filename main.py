import cv2
import numpy as np

image = cv2.imread(r"C:\Users\Michal\Documents\code\python\raspi\PK.png")

szary_obraz = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)           #konwertowanie na grayscale
odwrocony_obraz = cv2.bitwise_not(szary_obraz)                  #Odwrócenie obrazu
rozmyty_obraz = cv2.GaussianBlur(odwrocony_obraz, (21, 21), 0)
odwrocony_rozmyty = cv2.bitwise_not(rozmyty_obraz)
szkic = cv2.divide(szary_obraz, odwrocony_rozmyty, scale=256.0) #konwertowanie obrazu na szkic

# Zapisywanie obrazu szkicu
cv2.imwrite(r"C:\Users\Michal\Documents\code\python\raspi\szkic.png", szkic)
cv2.imshow("Obraz szkicu", szkic)
cv2.waitKey(0)
cv2.destroyAllWindows()

#znak wodny

logo = cv2.imread(r"C:\Users\Michal\Documents\code\python\raspi\szkic.png")
img = cv2.imread(r"C:\Users\Michal\Documents\code\python\raspi\background.jpg")

# wymiary obrazu znaku wodnego
h_logo, w_logo, _ = logo.shape

#Wymiary obrazu
h_img, w_img, _ = img.shape

# Obliczanie współrzędnych środka, w którym ma zostać umieszczony znak wodny
center_y = int(h_img/2)
center_x = int(w_img/2)

# obliczanie współrzędnych górnej, dolnej, lewej i prawej krawędzi
top_y = center_y - int(h_logo/2)
left_x = center_x - int(w_logo/2)
bottom_y = top_y + h_logo
right_x = left_x + w_logo

#Dodawanie znaku wodnego do obrazu
destination = img[top_y:bottom_y, left_x:right_x]
result = cv2.addWeighted(destination, 1, logo, 0.5, 0)

#Umieszczanie obszaru ze znakiem wodnym w obrazie
img[top_y:bottom_y, left_x:right_x] = result

#Zapisywanie i wyświetlanie obrazu z znakiem wodnym
cv2.imwrite(r"C:\Users\Michal\Documents\code\python\raspi\watermarked.jpg", img)
cv2.imshow("Obraz ze znakiem wodnym", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#definiowanie zmiennych do przechowywania współrzędnych, w których ma zostać umieszczony drugi obraz
positions=[]
positions2=[]
count=0

#funkcja do zapisywania współrzędnych kliknięć myszy
def draw_circle(event,x,y,flags,param):
    global positions,count
    # jeśli zdarzenie to kliknięcie lewym przyciskiem myszy, to zapisz współrzędne w listach positions i positions2
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(building,(x,y),2,(255,0,0),-1)
        positions.append([x,y])
        if(count!=3):
            positions2.append([x,y])
        elif(count==3):
            positions2.insert(2,[x,y])
        count+=1
        

building = cv2.imread(r'C:\Users\Michal\Documents\code\python\raspi\tablica.jpg')
dp = cv2.imread(r'C:\Users\Michal\Documents\code\python\raspi\watermarked.jpg')

# Tworzenie okna o nazwie 'image'
cv2.namedWindow('image')

cv2.setMouseCallback('image',draw_circle)

while(True):
    cv2.imshow('image',building)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

height, width = building.shape[:2]
h1,w1 = dp.shape[:2]

pts1=np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])
pts2=np.float32(positions)


h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)

height, width, channels = building.shape
im1Reg = cv2.warpPerspective(dp, h, (width, height))

mask2 = np.zeros(building.shape, dtype=np.uint8)

roi_corners2 = np.int32(positions2)

channel_count2 = building.shape[2]  
ignore_mask_color2 = (255,)*channel_count2

cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

mask2 = cv2.bitwise_not(mask2)
masked_image2 = cv2.bitwise_and(building, mask2)

#uzycie operatora Bitwise OR do połączenia obu obrazów
final = cv2.bitwise_or(im1Reg, masked_image2)
cv2.imwrite(r'C:\Users\Michal\Documents\code\python\raspi\final.png',final)
