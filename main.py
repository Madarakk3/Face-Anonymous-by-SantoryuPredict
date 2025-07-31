import os
import argparse
import cv2
from mediapipe.python.solutions.face_detection import FaceDetection
import mediapipe as mp

def process_img(img, face_detection):                                    # функция с аргументами изображение и объект детекции

    H, W, _ = img.shape                                                  # выводим height width и пропускаем каналы _

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       # выводим изображение в RGB потому что mediapipe работает с RGB
    out = face_detection.process(img_rgb)                                # сохраняем результат нашего объекта в out
                                                                         # .process метод который вызывает саму модель которую мы потом вставим в переменную

    if out.detections is not None:                                       # .process выдает результат в .detections либо гуд то есть лицо найдено либо None
                                                                         # ставим условия что если не будет None то запускаем цикл
        for detection in out.detections:                                 # делаем цикл по все лицам в out.detections
            location_data = detection.location_data                      # location_data содержит все координаты
            bbox = location_data.relative_bounding_box                   # relative_bounding_box хранит все прямоугольники вокруг лица соотвественно bbox

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height # даем поочередно нашим переменным их значение x1 = xmin и тд

            x1 = int(x1 * W)                                             # переводим координаты в пиксели умнажая их на ширину и высоту
                                                                         # для чего мы умножаем на ширину и высоту? потому что mediapipe выдает нам
            y1 = int(y1 * H)                                                  # значение в 0.3/0.1/0.2 и тд умножая эти значение на ширину и высоту
                                                                              # мы получим точные координаты лица для OpenCV
            w = int(w * W)
            h = int(h * H)

            #print(x1, y1, w, h)

            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))
                                                                         # мы берем координаты лица и размываем его k size = (30,30) то есть уровень размытия
                                        #берем исходное изображение и даем в него координаты лица тоесть от y1 до y1+высота и тд, канали ставім по нулям :
                                                                         # делаем блюр и добавляем те же координаты + ksize
    return img                           # возвращаем исходное изображение


args = argparse.ArgumentParser()         # объект парсера аргументов, который будет "собирать" параметры, переданные при запуске Python-скрипта.

args.add_argument("--mode", default='webcam') # Добавляем первый аргумент называем "--mode", если пользователь ничего не указал = "webcam"
args.add_argument("--filePath", default=None) # Добавляем второй аргумент называем "--filePath", сохраняем файл в этом аргументе и
                                                                                                                    # если пользователь ничего не указал = None

args = args.parse_args() # превращаем в объект с аргументами, где мы можем получить доступ вот так: # args.mode        #'webcam' или 'file'
                                                                                                    # args.filePath   # путь к файлу или None



output_dir = './output'                #output_dir — переменная, в которой хранится строка с путём до папки
                                       # куда вы будете сохранять результаты (например, обработанные изображения, логи и т.д.).
                                       # './output' означает:. — текущая рабочая директория скрипта и /output — подпапка output внутри неё
if not os.path.exists(output_dir):     # os.path.exists(path) — функция из модуля os.path, возвращающая:
                                       # True, если по указанному path уже есть файл или папка
os.makedirs(output_dir)                # False, если его нет и При помощи not мы инвертируем результат:
                                       # Условие True только когда папки ещё нет значит, нужно её создать

# os.makedirs(path) — функция из модуля os, создаёт все необходимые вложенные папки по указанному пути.
# В отличие от os.mkdir(), makedirs создаст сразу весь путь, даже если нужны несколько уровней вложенности (например, a/b/c).

# detect faces
mp_face_detection = mp.solutions.face_detection # для удобства создаем переменую и вставляем в нее наш выбор модели face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Создаем нашу модель facedetection и даем параметры (модель 0 для близкого контакта с камерой и моедль 1 для дальнего в полный рост)
    #  min_detection_confidence=0.5 даем увереность модели в 50 процентов
    # выводим наш параметр для нашей функции process_img с помощью as face_detection

    if args.mode in ["image"]:                     #«Посмотри, какой режим (mode) я передал при запуске скрипта.
                                                                # Если я указал --mode image (то есть args.mode == "image"), то продолжаем блок кода
        # read image
        img = cv2.imread(args.filePath)            # Читаем из файла (путь берётся из args.filePath) — получаем img как матрицу BGR-пикселей.

        img = process_img(img, face_detection)     #Вызываем функцию process_img она вернёт то же изображение, но с размытыми лицами.

        # save image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)   # Сохраняем результат в файл ./output/output.png.
                                                                   #os.path.join(output_dir, 'output.png') — корректно формирует путь в любой ОС.

    elif args.mode in ['video']:                             # «Посмотри, какой режим (mode) я передал при запуске скрипта.
                                                             # Если я указал --mode video (то есть args.mode == "video"), то продолжаем блок кода

        cap = cv2.VideoCapture(args.filePath)               # пусть к видео в args.filePath
        ret, frame = cap.read()                             # cap.read выдает 2 значение успешность чтение видео и самое видео

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),   # Создаём объект для записи видео:
                                       cv2.VideoWriter_fourcc(*'MP4V'),          # Путь и имя выходного файла (output/output.mp4).
                                       25,                                       # fourcc — кодек MP4V (четырёхсимвольный код).
                                       (frame.shape[1], frame.shape[0]))         # 25 — частота кадров (FPS).
                                                                                 # (width, height) — размеры кадра (из frame.shape: [0]=H, [1]=W).



        while ret:                                               # цикл до тех пор пока кадр считывается

            frame = process_img(frame, face_detection)           # выводим нашу фукнцию

            output_video.write(frame)                            # Записываем изменённый кадр в выходное видео.

            ret, frame = cap.read()                              # cap.read выдает 2 значение успешность чтение видео и самое видео

        cap.release()                                            # cap.release() — закрываем входное видео.
        output_video.release()                                   # #output_video.release() — сохраняем и закрываем выходной файл.


    elif args.mode in ['webcam']:                                #«Посмотри, какой режим (mode) я передал при запуске скрипта.
                                                                 # Если я указал --mode webcam (то есть args.mode == "webcam"), то продолжаем блок кода
        cap = cv2.VideoCapture(0)                                # считываем веб камеру у меня по дефолту 0

        ret, frame = cap.read()                                  # # cap.read выдает 2 значение успешность чтение видео и сам видео поток
        while ret:                                               # цикл до тех пор пока читается видео
            frame = process_img(frame, face_detection)           # выводим нашу фукнцию

            cv2.imshow('frame', frame)                  # показываем окно с результатом
            cv2.waitKey(25)                                     # ждем 25 мс и читаем следующий кадр

            ret, frame = cap.read()

        cap.release()                                          # освобождаем камеру после цикла