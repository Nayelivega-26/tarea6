import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo de lenguaje de señas previamente entrenado
model = tf.keras.models.load_model('modelo_lenguaje_senas.h5')

# Lista de nombres de clases (asegúrate de que coincidan con las salidas de tu modelo)
nombres_clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                  'U', 'V', 'W', 'X', 'Y', 'Z']

# Función para predecir el gesto a partir de los landmarks
def predecir_gesto(frame, landmarks):
    altura, ancho, _ = frame.shape
    puntos = []

    for i in range(21):  # 21 puntos de la mano
        x = int(landmarks.landmark[i].x * ancho)
        y = int(landmarks.landmark[i].y * altura)
        puntos.append([x, y])

    puntos = np.array(puntos)
    puntos = puntos.flatten() / 255.0  # Normalizar

    # Asegúrate que tu modelo fue entrenado con esta entrada (42 valores: 21 x, y)
    puntos = puntos.reshape(1, -1)  # Redimensionar para que sea 1 muestra de entrada

    prediccion = model.predict(puntos)
    clase = np.argmax(prediccion)

    return clase

# Función para mostrar la letra predicha en la pantalla
def mostrar_clase(frame, clase):
    texto = f'Letra: {nombres_clases[clase]}'
    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

# Verificar si la cámara está funcionando
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara abierta correctamente. Presiona 'q' para salir.")

# Iniciar el bucle de captura de frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar el frame.")
        break

    # Voltear horizontalmente
    frame = cv2.flip(frame, 1)

    # Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe
    resultados = hands.process(frame_rgb)

    if resultados.multi_hand_landmarks:
        for landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Predecir la clase del gesto
            clase = predecir_gesto(frame, landmarks)

            # Mostrar resultado
            mostrar_clase(frame, clase)

    # Mostrar ventana
    cv2.imshow("Detección de Lenguaje de Señas", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
