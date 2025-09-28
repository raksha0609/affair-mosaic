# scripts/phantoms.py
"""
Generate synthetic evaluation images (phantoms) with geometric primitives.
Produces: data/phantoms/target_*.png and data/phantoms/tiles/*.png
"""
import os, cv2, numpy as np, random
os.makedirs("data/phantoms/tiles", exist_ok=True)
os.makedirs("data/phantoms/targets", exist_ok=True)

def make_tile(w=200,h=200, seed=0):
    random.seed(seed)
    img = np.ones((h,w,3), dtype='uint8')*255
    # random background texture
    for _ in range(400):
        x = random.randint(0,w-1); y = random.randint(0,h-1)
        img[y,x] = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
    # draw primitives
    for _ in range(random.randint(1,4)):
        c = tuple(random.randint(0,255) for _ in range(3))
        r = random.randint(10,60)
        cx = random.randint(r,w-r); cy = random.randint(r,h-r)
        cv2.circle(img, (cx,cy), r, c, -1)
    for _ in range(random.randint(0,3)):
        x0,y0 = random.randint(0,w-1), random.randint(0,h-1)
        x1,y1 = random.randint(0,w-1), random.randint(0,h-1)
        cv2.line(img, (x0,y0),(x1,y1), tuple(random.randint(0,255) for _ in range(3)), thickness=random.randint(1,4))
    return img

# create tiles
for k in range(100):
    t = make_tile(200,200, seed=k)
    cv2.imwrite(f"data/phantoms/tiles/tile_{k:03d}.png", t)

# create a target composed of grid of tiles with random transforms
grid = 6
tile_files = [f"data/phantoms/tiles/{x}" for x in sorted(os.listdir("data/phantoms/tiles"))[:100]]
out = np.zeros((grid*200, grid*200, 3), dtype='uint8')
for i in range(grid):
    for j in range(grid):
        idx = random.randint(0,len(tile_files)-1)
        t = cv2.imread(tile_files[idx])
        # random transform: rotate, crop, zoom
        ang = random.choice([0,90,180,270])
        t2 = cv2.rotate(t, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])) if random.random()<0.4 else t
        # slight brightness change
        t2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
        t2 = np.clip(t2.astype('int32') + np.array([random.randint(-20,20),0,0]), 0,255).astype('uint8')
        t2 = cv2.cvtColor(t2, cv2.COLOR_HSV2BGR)
        out[i*200:(i+1)*200, j*200:(j+1)*200] = t2
cv2.imwrite("data/phantoms/targets/phantom_target.png", out)
print("Phantoms generated in data/phantoms/")
