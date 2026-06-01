"""
Generate synthetic CT DICOM files for various human body parts.
Each body part gets a unique geometric pattern that roughly
suggests its anatomical shape when viewed as a cross-section.

Usage:
    python samples/generate.py [--list] [--all] [--body-part NAME]
"""
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage
import sys

SIZE = 128

BODY_PARTS = {
    'brain':      ('Oval with ventricles', lambda n: _brain(n)),
    'head':       ('Oval with skull ring', lambda n: _head(n)),
    'neck':       ('Circular with spine', lambda n: _neck(n)),
    'chest':      ('Lungs and heart', lambda n: _chest(n)),
    'heart':      ('Heart-shaped', lambda n: _heart(n)),
    'lung':       ('Lobe pairs', lambda n: _lung(n)),
    'breast':     ('Soft tissue mass', lambda n: _breast(n)),
    'abdomen':    ('Liver and organs', lambda n: _abdomen(n)),
    'liver':      ('Wedge shape', lambda n: _liver(n)),
    'kidney':     ('Bean shape', lambda n: _kidney(n)),
    'stomach':    ('Crescent shape', lambda n: _stomach(n)),
    'spleen':     ('Oval organ', lambda n: _spleen(n)),
    'pancreas':   ('Elongated shape', lambda n: _pancreas(n)),
    'intestine':  ('Coiled tubes', lambda n: _intestine(n)),
    'spine':      ('Vertebra cross', lambda n: _spine(n)),
    'pelvis':     ('Pelvic ring', lambda n: _pelvis(n)),
    'hip':        ('Joint socket', lambda n: _hip(n)),
    'shoulder':   ('Joint with humerus', lambda n: _shoulder(n)),
    'arm':        ('Circular bone', lambda n: _arm(n)),
    'elbow':      ('Joint shape', lambda n: _elbow(n)),
    'forearm':    ('Dual bones', lambda n: _forearm(n)),
    'wrist':      ('Multiple small bones', lambda n: _wrist(n)),
    'hand':       ('Palm and fingers', lambda n: _hand(n)),
    'finger':     ('Small oval', lambda n: _finger(n)),
    'rib':        ('Curved bone', lambda n: _rib(n)),
    'clavicle':   ('S-curve bone', lambda n: _clavicle(n)),
    'scapula':    ('Flat triangle', lambda n: _scapula(n)),
    'thigh':      ('Large bone circle', lambda n: _thigh(n)),
    'femur':      ('Bone cross-section', lambda n: _femur(n)),
    'knee':       ('Patella and condyles', lambda n: _knee(n)),
    'calf':       ('Tibia and fibula', lambda n: _calf(n)),
    'ankle':      ('Talus and malleoli', lambda n: _ankle(n)),
    'foot':       ('Arch and metatarsals', lambda n: _foot(n)),
    'toe':        ('Small phalanx', lambda n: _toe(n)),
    'sinus':      ('Air-filled cavities', lambda n: _sinus(n)),
    'jaw':        ('Mandible U-shape', lambda n: _jaw(n)),
    'tooth':      ('Enamel and root', lambda n: _tooth(n)),
    'eye':        ('Orbit with optic nerve', lambda n: _eye(n)),
    'ear':        ('Auditory canal', lambda n: _ear(n)),
    'thyroid':    ('Butterfly shape', lambda n: _thyroid(n)),
    'trachea':    ('Ringed airway', lambda n: _trachea(n)),
    'bladder':    ('Balloon shape', lambda n: _bladder(n)),
    'prostate':   ('Walnut shape', lambda n: _prostate(n)),
    'uterus':     ('Pear shape', lambda n: _uterus(n)),
    'ovary':      ('Almond shape', lambda n: _ovary(n)),
    'testicle':   ('Oval paired', lambda n: _testicle(n)),
    'muscle':     ('Fiber bundle', lambda n: _muscle(n)),
    'artery':     ('Vessel cross-section', lambda n: _artery(n)),
    'vein':       ('Thin-walled vessel', lambda n: _vein(n)),
    'lymph':      ('Small oval node', lambda n: _lymph(n)),
    'fat':        ('Adipose tissue', lambda n: _fat(n)),
    'skin':       ('Layered tissue', lambda n: _skin(n)),
    'bone':       ('Dense cortical', lambda n: _bone(n)),
    'cartilage':  ('Smooth joint surface', lambda n: _cartilage(n)),
    'tendon':     ('Dense fiber band', lambda n: _tendon(n)),
    'nerve':      ('Bundle of fibers', lambda n: _nerve(n)),
    'diaphragm':  ('Domed muscle', lambda n: _diaphragm(n)),
    'aorta':      ('Large vessel', lambda n: _aorta(n)),
    'esophagus':  ('Muscular tube', lambda n: _esophagus(n)),
}


def _gaussian_blur(arr, sigma=2):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr.astype(float), sigma)


def _normalize(arr, lo=0, hi=200):
    arr = arr.astype(float)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * (hi - lo) + lo
    return arr.astype(np.uint16)


def _circle(n, r, cx=None, cy=None):
    yy, xx = np.ogrid[:n, :n]
    cx = cx or (n - 1) / 2
    cy = cy or (n - 1) / 2
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= r


def _ellipse(n, rx, ry, cx=None, cy=None):
    yy, xx = np.ogrid[:n, :n]
    cx = cx or (n - 1) / 2
    cy = cy or (n - 1) / 2
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def _ring(n, r_outer, r_inner):
    return _circle(n, r_outer) & ~_circle(n, r_inner)


def _brain(n):
    m = _ellipse(n, 35, 45) ^ _ellipse(n, 15, 20)
    m = m.astype(float)
    # sulci pattern
    for _ in range(40):
        x, y = np.random.randint(n//4, 3*n//4, 2)
        m[max(0,y-1):min(n,y+2), max(0,x-1):min(n,x+2)] = 0.3
    # ventricles
    m[_ellipse(n, 6, 10, n//2-8, n//2)] -= 0.5
    m[_ellipse(n, 6, 10, n//2+8, n//2)] -= 0.5
    return _normalize(m)


def _head(n):
    m = _ellipse(n, 38, 48).astype(float) * 100
    m[_ellipse(n, 33, 43)] = 150  # brain
    m[_ring(n, 38, 36)] = 250  # skull
    return _normalize(_gaussian_blur(m, 1))


def _neck(n):
    m = _circle(n, 22).astype(float) * 100
    m[_circle(n, 4, n//2, n//2+6)] = 250  # spine
    m[_ellipse(n, 12, 5, n//2, n//2-10)] = 80  # trachea
    m[_ellipse(n, 4, 8, n//2-10, n//2)] = 120  # carotid
    m[_ellipse(n, 4, 8, n//2+10, n//2)] = 120  # carotid
    return _normalize(_gaussian_blur(m, 1))


def _chest(n):
    m = _ellipse(n, 40, 50).astype(float) * 100  # body
    m[_ellipse(n, 18, 22, n//2, n//2-8)] = 200  # heart
    m[_ellipse(n, 22, 16, n//2+8, n//2+4)] = 40  # right lung
    m[_ellipse(n, 22, 16, n//2-8, n//2+4)] = 40  # left lung
    m[_circle(n, 3, n//2, n//2+6)] = 250  # spine
    return _normalize(_gaussian_blur(m, 1))


def _heart(n):
    m = np.zeros((n, n))
    h = _ellipse(n, 18, 22)
    m[h] = 180
    m[_ellipse(n, 10, 14, n//2, n//2+6)] = 120
    m[_ellipse(n, 8, 10, n//2-4, n//2-4)] = 120
    return _normalize(_gaussian_blur(m, 2))


def _lung(n):
    m = np.ones((n, n)) * 40
    m[_ellipse(n, 30, 20, n//2-6, n//2)] = 180  # lobes
    m[_ellipse(n, 30, 20, n//2+6, n//2)] = 180
    return _normalize(_gaussian_blur(m, 1))


def _breast(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 30, 20, n//2, n//2+6)] = 150
    m[_ellipse(n, 30, 20, n//2-24, n//2+6)] = 150  # pair
    return _normalize(_gaussian_blur(m, 2))


def _abdomen(n):
    m = _ellipse(n, 40, 35).astype(float) * 80
    m[_ellipse(n, 18, 12, n//2+6, n//2)] = 140  # liver
    m[_ellipse(n, 12, 14, n//2-8, n//2-2)] = 100  # spleen
    m[_circle(n, 3, n//2, n//2+10)] = 250  # spine
    m[_ellipse(n, 8, 5, n//2-4, n//2-8)] = 60  # bowel
    return _normalize(_gaussian_blur(m, 1))


def _liver(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 25, 18)] = 140
    m[_circle(n, 8, n//2+12, n//2-4)] = 120
    return _normalize(_gaussian_blur(m, 2))


def _kidney(n):
    m = np.zeros((n, n))
    b = _ellipse(n, 12, 20)
    m[b] = 130
    # bean indent
    m[_circle(n, 4, n//2+10, n//2)] = 60
    m[_circle(n, 4, n//2-10, n//2)] = 60
    return _normalize(_gaussian_blur(m, 1))


def _stomach(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 20, 14, n//2+4, n//2)] = 100
    m[_ellipse(n, 14, 10, n//2-6, n//2-2)] = 60
    return _normalize(_gaussian_blur(m, 2))


def _spleen(n):
    return _normalize(_gaussian_blur(_ellipse(n, 14, 22, n//2-4, n//2).astype(float) * 120, 2))


def _pancreas(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 20, 8)] = 110
    m[_ellipse(n, 8, 5, n//2+16, n//2+2)] = 110
    return _normalize(_gaussian_blur(m, 1))


def _intestine(n):
    m = np.ones((n, n)) * 50
    for i in range(6):
        cx, cy = np.random.randint(15, n-15, 2)
        r = np.random.randint(6, 12)
        m[_circle(n, r, cx, cy)] = 80
        m[_circle(n, r-3, cx, cy)] = 40
    return _normalize(_gaussian_blur(m, 1))


def _spine(n):
    m = np.zeros((n, n))
    m[_circle(n, 10)] = 250  # vertebral body
    m[_ellipse(n, 18, 6)] = 200  # transverse processes
    m[_ellipse(n, 6, 18)] = 200
    m[_circle(n, 2)] = 100  # spinal canal
    return _normalize(m)


def _pelvis(n):
    m = np.zeros((n, n))
    m[_ring(n, 35, 28)] = 220  # pelvic ring
    m[_circle(n, 5, n//2-14, n//2)] = 250  # acetabulum
    m[_circle(n, 5, n//2+14, n//2)] = 250
    m[_circle(n, 4, n//2, n//2+20)] = 250  # sacrum
    return _normalize(_gaussian_blur(m, 1))


def _hip(n):
    m = np.zeros((n, n))
    m[_circle(n, 16, n//2-6, n//2)] = 220  # femoral head
    m[_circle(n, 14, n//2+6, n//2)] = 200  # acetabulum
    m[_circle(n, 4, n//2-6, n//2-10)] = 250  # trochanter
    return _normalize(_gaussian_blur(m, 1))


def _shoulder(n):
    m = np.zeros((n, n))
    m[_circle(n, 12, n//2+10, n//2-4)] = 250  # humeral head
    m[_ellipse(n, 20, 8, n//2-8, n//2-4)] = 200  # glenoid
    return _normalize(_gaussian_blur(m, 1))


def _arm(n):
    m = np.zeros((n, n))
    m[_circle(n, 10)] = 220  # humerus
    m[_ring(n, 16, 11)] = 100  # muscle
    return _normalize(_gaussian_blur(m, 1))


def _elbow(n):
    m = np.zeros((n, n))
    m[_circle(n, 8, n//2-4, n//2-4)] = 240
    m[_circle(n, 7, n//2+4, n//2+4)] = 240
    return _normalize(_gaussian_blur(m, 1))


def _forearm(n):
    m = np.zeros((n, n))
    m[_circle(n, 5, n//2-5, n//2)] = 230  # radius
    m[_circle(n, 6, n//2+5, n//2)] = 230  # ulna
    return _normalize(_gaussian_blur(m, 1))


def _wrist(n):
    m = np.zeros((n, n))
    for dx, dy in [(-6,-4), (0,-6), (6,-4), (-4,2), (4,2), (-2,6), (2,6)]:
        m[_circle(n, 4, n//2+dx, n//2+dy)] = 220
    return _normalize(_gaussian_blur(m, 1))


def _hand(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 22, 14)] = 100  # palm
    for dx in [-12, -4, 4, 12]:
        m[_ellipse(n, 3, 8, n//2+dx, n//2-16)] = 200  # fingers
    m[_ellipse(n, 4, 6, n//2, n//2+12)] = 200  # thumb
    return _normalize(_gaussian_blur(m, 1))


def _finger(n):
    return _normalize(_gaussian_blur(_ellipse(n, 5, 12).astype(float) * 220, 1))


def _rib(n):
    m = np.zeros((n, n))
    yy, xx = np.ogrid[:n, :n]
    cx, cy = n//2-10, n//2+10
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    m[(r > 22) & (r < 28) & (xx > cx)] = 200  # curved bone
    return _normalize(_gaussian_blur(m, 1))


def _clavicle(n):
    m = np.zeros((n, n))
    yy, xx = np.ogrid[:n, :n]
    cx, cy = n//2, n//2+10
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    m[(r > 15) & (r < 18)] = 220
    return _normalize(m)


def _scapula(n):
    return _normalize(_gaussian_blur(_ellipse(n, 26, 10, n//2-4, n//2).astype(float) * 200, 2))


def _thigh(n):
    m = np.zeros((n, n))
    m[_circle(n, 10)] = 240  # femur
    m[_ring(n, 22, 11)] = 100  # muscle
    m[_ring(n, 25, 23)] = 60  # fat
    return _normalize(_gaussian_blur(m, 1))


def _femur(n):
    m = np.zeros((n, n))
    m[_circle(n, 12)] = 250  # cortical bone
    m[_circle(n, 6)] = 100  # marrow
    return _normalize(m)


def _knee(n):
    m = np.zeros((n, n))
    m[_circle(n, 10, n//2-6, n//2+4)] = 240  # medial condyle
    m[_circle(n, 9, n//2+6, n//2+4)] = 240  # lateral condyle
    m[_circle(n, 6, n//2, n//2-8)] = 200  # patella
    return _normalize(_gaussian_blur(m, 1))


def _calf(n):
    m = np.zeros((n, n))
    m[_circle(n, 4, n//2-4, n//2)] = 230  # tibia
    m[_circle(n, 3, n//2+6, n//2+2)] = 210  # fibula
    m[_ring(n, 16, 5)] = 100  # muscle
    return _normalize(_gaussian_blur(m, 1))


def _ankle(n):
    m = np.zeros((n, n))
    m[_circle(n, 6, n//2, n//2+2)] = 240  # talus
    m[_circle(n, 4, n//2-6, n//2-4)] = 220  # medial malleolus
    m[_circle(n, 4, n//2+6, n//2-4)] = 220  # lateral malleolus
    return _normalize(_gaussian_blur(m, 1))


def _foot(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 18, 8)] = 80  # arch
    m[_ellipse(n, 4, 8, n//2-6, n//2+6)] = 200  # metatarsals
    m[_ellipse(n, 4, 8, n//2+6, n//2+6)] = 200
    m[_ellipse(n, 3, 6, n//2, n//2+8)] = 200
    m[_circle(n, 5, n//2, n//2-6)] = 240  # calcaneus
    return _normalize(_gaussian_blur(m, 1))


def _toe(n):
    return _normalize(_gaussian_blur(_ellipse(n, 4, 8).astype(float) * 220, 1))


def _sinus(n):
    m = np.ones((n, n)).astype(float) * 50
    m[_ellipse(n, 8, 12, n//2-8, n//2-6)] = 10  # maxillary
    m[_ellipse(n, 8, 12, n//2+8, n//2-6)] = 10
    m[_ellipse(n, 4, 4, n//2, n//2-18)] = 10  # frontal
    return _normalize(m)


def _jaw(n):
    m = np.zeros((n, n))
    m[_ring(n, 24, 16)] = 240
    m[_circle(n, 3, n//2-14, n//2)] = 200  # teeth
    m[_circle(n, 3, n//2+14, n//2)] = 200
    return _normalize(_gaussian_blur(m, 1))


def _tooth(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 6, 10)] = 250  # enamel
    m[_ellipse(n, 3, 6)] = 150  # pulp
    m[_ellipse(n, 4, 5, n//2, n//2+10)] = 200  # root
    return _normalize(m)


def _eye(n):
    m = np.zeros((n, n))
    m[_circle(n, 10)] = 150  # globe
    m[_circle(n, 4, n//2-4, n//2)] = 50  # lens
    m[_ellipse(n, 2, 6, n//2+10, n//2)] = 100  # optic nerve
    return _normalize(_gaussian_blur(m, 1))


def _ear(n):
    m = np.zeros((n, n))
    m[_ring(n, 12, 8)] = 200
    m[_circle(n, 3, n//2-4, n//2-6)] = 50  # canal
    return _normalize(_gaussian_blur(m, 1))


def _thyroid(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 10, 14, n//2-6, n//2)] = 140
    m[_ellipse(n, 10, 14, n//2+6, n//2)] = 140
    m[_ellipse(n, 4, 3, n//2, n//2-10)] = 110  # isthmus
    return _normalize(_gaussian_blur(m, 1))


def _trachea(n):
    m = np.ones((n, n)) * 60
    m[_ring(n, 12, 10)] = 200  # cartilage ring
    m[_circle(n, 9)] = 20  # airway
    return _normalize(m)


def _bladder(n):
    m = np.zeros((n, n))
    m[_circle(n, 15)] = 120  # wall
    m[_circle(n, 12)] = 60  # fluid
    return _normalize(_gaussian_blur(m, 2))


def _prostate(n):
    return _normalize(_gaussian_blur(_circle(n, 10).astype(float) * 130, 2))


def _uterus(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 12, 18)] = 130
    m[_ellipse(n, 8, 6, n//2, n//2-14)] = 110  # cervix
    return _normalize(_gaussian_blur(m, 2))


def _ovary(n):
    return _normalize(_gaussian_blur(_ellipse(n, 8, 12, n//2-4, n//2).astype(float) * 120, 1))


def _testicle(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 6, 10, n//2-5, n//2)] = 100
    m[_ellipse(n, 6, 10, n//2+5, n//2)] = 100
    return _normalize(_gaussian_blur(m, 1))


def _muscle(n):
    m = np.zeros((n, n))
    for i in range(8):
        m[_ellipse(n, 4, 14, n//2, n//2-14+i*4)] = 100
    return _normalize(_gaussian_blur(m, 1))


def _artery(n):
    m = np.zeros((n, n))
    m[_ring(n, 10, 8)] = 200  # wall
    m[_circle(n, 7)] = 80  # lumen
    return _normalize(m)


def _vein(n):
    m = np.zeros((n, n))
    m[_ring(n, 12, 10)] = 120  # thin wall
    m[_circle(n, 9)] = 60  # lumen
    return _normalize(m)


def _lymph(n):
    return _normalize(_gaussian_blur(_ellipse(n, 5, 8).astype(float) * 120, 1))


def _fat(n):
    return _normalize(np.ones((n, n)) * 60 + np.random.randn(n, n) * 5)


def _skin(n):
    m = np.ones((n, n)) * 80
    m[_ring(n, 30, 28)] = 120  # dermis
    m[_ring(n, 28, 26)] = 60  # subcutaneous
    return _normalize(m)


def _bone(n):
    m = np.zeros((n, n))
    m[_ring(n, 14, 8)] = 250
    m[_circle(n, 7)] = 120
    return _normalize(m)


def _cartilage(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 12, 8)] = 180
    return _normalize(_gaussian_blur(m, 2))


def _tendon(n):
    m = np.zeros((n, n))
    m[_ellipse(n, 4, 16)] = 220
    return _normalize(_gaussian_blur(m, 1))


def _nerve(n):
    m = np.zeros((n, n))
    for i in range(5):
        m[_circle(n, 2, n//2, n//2-12+i*6)] = 150
    return _normalize(_gaussian_blur(m, 1))


def _diaphragm(n):
    m = np.ones((n, n)) * 60
    yy, xx = np.mgrid[:n, :n]
    m[yy < 0.3*n] = 40
    m[yy > 0.7*n] = 80
    m[(yy > 0.3*n) & (yy < 0.7*n)] = 140
    return _normalize(_gaussian_blur(m, 3))


def _aorta(n):
    m = np.zeros((n, n))
    m[_ring(n, 12, 9)] = 200
    m[_circle(n, 8)] = 70
    return _normalize(m)


def _esophagus(n):
    m = np.zeros((n, n))
    m[_ring(n, 6, 4)] = 130
    m[_circle(n, 3)] = 40
    return _normalize(m)


def make_ct_dicom(pixels, name, desc):
    n = pixels.shape[0]
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = f'Synthetic^{name}'
    ds.PatientID = f'SYN-{name}'
    ds.PatientSex = 'O'
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = '1'
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.Modality = 'CT'
    ds.Manufacturer = 'CT Reconstruction Project'
    ds.SeriesDescription = f'Synthetic {name} cross-section'
    ds.Rows = n
    ds.Columns = n
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024
    ds.RescaleType = 'HU'
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.SamplesPerPixel = 1
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.PixelData = (pixels.astype(np.int16) + 1024).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    path = f'samples/{name}.dcm'
    ds.save_as(path)
    print(f'  [OK] {name}.dcm')


def generate(name):
    name = name.lower().replace(' ', '-')
    if name not in BODY_PARTS:
        print(f'Unknown body part: {name}')
        print(f'Available: {", ".join(sorted(BODY_PARTS.keys()))}')
        return False
    label, func = BODY_PARTS[name]
    arr = func(SIZE)
    make_ct_dicom(arr, name, f'{label} (synthetic)')
    return True


def generate_all():
    print(f'Generating {len(BODY_PARTS)} synthetic DICOM files...')
    for name in sorted(BODY_PARTS.keys()):
        generate(name)
    print(f'\nDone. {len(BODY_PARTS)} files in samples/')
    print(f'Total size: ~{len(BODY_PARTS) * 33} KB')


def list_parts():
    print(f'Available body parts ({len(BODY_PARTS)}):')
    for name, (label, _) in sorted(BODY_PARTS.items()):
        print(f'  {name:18s} {label}')


if __name__ == '__main__':
    if '--list' in sys.argv:
        list_parts()
    elif '--all' in sys.argv:
        generate_all()
    elif len(sys.argv) > 1:
        for name in sys.argv[1:]:
            if name in ('--list', '--all'):
                continue
            generate(name)
    else:
        print('Usage: python samples/generate.py [--list|--all|body_part1 body_part2 ...]')
        print(f'Example: python samples/generate.py brain chest knee')
