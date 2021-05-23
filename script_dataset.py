import os
import requests
import sys
import cv2
import tqdm
from skimage import io

def dataset_creation(key="", img_num=50):

  page = 1
  imgbase_path = "https://www.themoviedb.org/t/p/w400/"
  test = requests.get(f"https://api.themoviedb.org/3/person/1?api_key={key}&language=en-US").status_code

  face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_frontalface_alt2.xml')

  if test != 200 :
    print('Invalid API key !')
    return 0
  
  if img_num < 1:
    print("The number of image should be at least greater than 1 !")
    return 0


  project_path = os.getcwd()
  woman_path = os.path.join(project_path, 'Female')
  man_path = os.path.join(project_path, 'Male')

  try:
    os.makedirs(woman_path)
    os.makedirs(man_path)
  except:
    pass

  with tqdm(total=100) as pbar:
    while (len(os.listdir(woman_path))<img_num or len(os.listdir(man_path))<img_num):

      elms = requests.get(f"https://api.themoviedb.org/3/person/popular?api_key={key}&language=en-US&page={page}").json()['results']

      for elm in elms:

        req = requests.get(f"https://api.themoviedb.org/3/person/{elm['id']}?api_key={key}&language=en-US")
        data = req.json()

        if req.status_code == 200 : 
          if data['profile_path']:

            img = io.imread(imgbase_path+data['profile_path'])
            
            try:
              img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
            except: #Means the image is already in grayscale
              faces = face_cascade.detectMultiScale(img, 1.1, 4)

            if len(faces)==1 :
              x, y, w, h = faces[0]
              crop_face = img[y:y + h, x:x + w]

              if int(data['gender']) == 1 and len(os.listdir(woman_path))<img_num:
                io.imsave(os.path.join(woman_path, f'{elm["id"]}.jpg'), crop_face)
                pbar.update((1/(2*img_num)) * 100)

              elif int(data['gender']) == 2 and len(os.listdir(man_path))<img_num:
                io.imsave(os.path.join(man_path, f'{elm["id"]}.jpg'), crop_face)
                pbar.update((1/(2*img_num)) * 100)

      page +=1
      if (len(os.listdir(woman_path))>=img_num and len(os.listdir(man_path))>=img_num): break
  

if __name__ =='__main__':
  if len(sys.argv) < 2 : dataset_creation()
  elif len(sys.argv) < 3 : dataset_creation(sys.argv[1])
  else : dataset_creation(sys.argv[1], int(sys.argv[2]))