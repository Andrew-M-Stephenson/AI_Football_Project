import cv2

def draw_tracks(frame, tracks, color=(0,255,0)):
    for t in tracks:
        x1,y1,x2,y2 = map(int, t["box"])
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f'ID {t["id"]}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame
