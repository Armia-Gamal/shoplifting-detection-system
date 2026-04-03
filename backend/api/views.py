from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
import numpy as np
import tempfile
from .model_loader import predict_sequence


@api_view(['POST'])
def predict(request):
    file = request.FILES.get('file')

    if not file:
        return Response({"error": "No file uploaded"}, status=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(file.read())
        temp_path = temp_video.name

    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        return Response({"error": "Cannot open video"}, status=400)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return Response({"error": "Empty or corrupted video"}, status=400)

    step = max(1, total_frames // 20)

    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frames.append(frame)

        if len(frames) == 20:
            break

    cap.release()

    while len(frames) < 20:
        frames.append(frames[-1])

    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)

    result = predict_sequence(frames)

    return Response({
        "prediction": result,
        "label": "Shoplifting" if result > 0.5 else "Normal"
    })