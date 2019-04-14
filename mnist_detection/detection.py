#Phần này dùng để đọc đường dẫn ảnh trong upload
from pathlib import Path
base_path = Path(__file__).parent
file_path = (base_path / "../server/upload/pj 204.png").resolve()
import matplotlib.image as mp
image_path = str(file_path)
image = mp.imread(image_path)
#end
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

