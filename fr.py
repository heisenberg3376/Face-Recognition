import face_recognition
import cv2
import os

KnownFacesDir = 'Known Faces'
UnknownFacesDir = 'Unknown Faces'
Tolerance = 0.7
FrameThickness = 3
FontThickness = 2
Model = 'cnn' #hog

print('loading known faces')

known_faces = []
known_names = []

# Known Faces > Khabib > [khb-1, khb-2, khb-3, ....]

for name in os.listdir(KnownFacesDir):
    for file in os.listdir(f'{KnownFacesDir}/{name}'):
        img = face_recognition.load_image_file(f'{KnownFacesDir}/{name}/{file}')
        encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(encoding)
        known_names.append(name)
        
print('processing Unknown faces')



for filename in os.listdir(UnknownFacesDir):
    print(filename)
    img = face_recognition.load_image_file(f'{UnknownFacesDir}/{filename}')
    img_locs = face_recognition.face_locations(img, model=Model)
    # returns location in TRBL format
    encodings = face_recognition.face_encodings(img, img_locs)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings, img_locs):
        results = face_recognition.compare_faces(known_faces, face_encoding, Tolerance)
        mtch = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'Found {match}')

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [255, 0, 0]

            cv2.rectangle(img, top_left, bottom_right, color, FrameThickness)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(img, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222,222,222), FontThickness)

        if False in results:
            nomatch = 'Not Found'

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 0, 255]

            cv2.rectangle(img, top_left, bottom_right, color, FrameThickness)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(img, nomatch, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222,222,222), FontThickness)
            

    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
