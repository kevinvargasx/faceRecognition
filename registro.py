import cv2
import os
import face_recognition
import csv
from datetime import datetime

# Codificar los rostros extraídos
imageFacesPath = "C:/Users/Usuario/Programando/IA_proyecto/faceRecognition/faces"

facesEncodings = []
facesNames = []
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(os.path.join(imageFacesPath, file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split(".")[0])

# Crear un archivo CSV para exportar los datos
csv_file = "rostros_reconocidos.csv"

# Detector facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

with open(csv_file, mode='w', newline='') as csv_file:
    fieldnames = ['Nombre', 'Fecha', 'Hora']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
    # Lista para almacenar los rostros previamente detectados
    rostros_previamente_detectados = []

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        orig = frame.copy()
        faces = faceClassif.detectMultiScale(frame, 1.1, 5)

        for (x, y, w, h) in faces:
            face = orig[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
            result = face_recognition.compare_faces(facesEncodings, actual_face_encoding)

            if True in result:
                index = result.index(True)
                name = facesNames[index]
                color = (125, 220, 0)

                # Verificar si el rostro no ha sido previamente detectado
                if name not in rostros_previamente_detectados:
                    # Guardar el nombre, fecha y hora en el archivo CSV
                    current_time = datetime.now()
                    writer.writerow({'Nombre': name, 'Fecha': current_time.date(), 'Hora': current_time.time()})
                    rostros_previamente_detectados.append(name)
            else:
                name = "Desconocido"
                color = (50, 50, 255)

            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
