import cv2
import numpy as np
import pyautogui
class HareketTakibi:
    def __init__(self):
        # Lucas-Kanade optik akis parametreleri
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # cilt rengi icin hsv belirtimi
        self.lower_color = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_color = np.array([20, 255, 255], dtype=np.uint8)

        self.capture = cv2.VideoCapture(0)
        ret, self.old_frame = self.capture.read()
        self.old_frame = cv2.flip(self.old_frame, 1)
        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = self.new_points(self.old_frame)

    def color_mask(self, frame):
        # cilt rengi icin maske olusumu
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        return mask
    def new_points(self, frame):
        mask = self.color_mask(frame)
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask, **dict(maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7))
        return p0
    def govde(self):
        # butun kareleri isleme ve fare kontrolu
        ret, frame = self.capture.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.color_mask(frame)

        if self.p0 is not None:
            # Optik akis
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]

                if len(good_new) > 0:
                    a, b = good_new[0].ravel().astype(int)
                    c, d = good_old[0].ravel().astype(int)
                    cv2.circle(frame, (a, b), 10, (0, 0, 255), thickness=cv2.FILLED)
                    dx, dy = a - c, b - d
                    pyautogui.moveRel(dx, dy)
                    self.old_gray = frame_gray.copy()
                    self.p0 = good_new.reshape(-1, 1, 2)
                else:
                    self.p0 = self.new_points(frame)
            else:
                self.p0 = self.new_points(frame)
        else:
            self.p0 = self.new_points(frame)
        return frame
    def running(self):
        while True:
            image = self.govde()
            if image is None:
                break
            cv2.imshow('optik_akis_hareket_takibi', image)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    control = HareketTakibi()
    control.running()