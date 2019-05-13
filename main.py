import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from PIL import Image

P_size = 500


class Shape:

    def __init__(self, shape_name, image_width, image_height, width, height, center, color, z):
        self.shape_name = shape_name
        self.image_width = image_width
        self.image_height = image_height
        self.width = width
        self.height = height
        self.center = center
        self.color = color
        self.z = z
        self.rr, self.cc = ellipse(self.width, self.height, self.center[0], self.center[1])

    def get_coordinates(self):
        rr, cc = ellipse(self.width, self.height, self.center[0], self.center[1])
        return rr, cc


def cross_over(shape1, shape2):
    child1 = Shape(shape1.shape_name, shape1.image_width, shape1.image_height, shape1.width, shape1.height,
                   shape1.center, shape2.color, shape2.z)
    child2 = Shape(shape2.shape_name, shape2.image_width, shape2.image_height, shape2.width, shape2.height,
                   shape2.center, shape1.color, shape1.z)
    return child1, child2


def mutate(shape, image_width, image_height):
    random_field = random.randint(0, 5)

    if random_field == 0:
        shape.width = random.randint(1, image_width)
    if random_field == 1:
        shape.height = random.randint(1, image_height)
    if random_field == 2:
        shape.center = (random.randint(0, image_width), random.randint(0, image_height))
    if random_field == 3:
        shape.color = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
    if random_field == 4:
        shape.z = random.uniform(0, 1)


def init_population(image_width, image_height):
    pop = []
    for i in range(0, P_size):
        shape_name = "ellipse"
        width = random.randint(1, image_width)
        height = random.randint(1, image_height)
        center = (random.randint(0, image_width), random.randint(0, image_height))
        color = [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]
        z = random.uniform(0, 1)
        shape = Shape(shape_name, image_width, image_height, width, height, center, color, z)
        pop.append(shape)
    return pop


def fitness(image, shape):
    score = 0.0
    rr, cc = shape.get_coordinates()

    for i in range(0, len(rr)):
        r = rr[i]
        c = cc[i]
        if 0 <= r < shape.image_width and 0 <= c < shape.image_height:
            pixel = image[r][c]
            score += np.sum(abs(pixel - shape.color))
    return score


def choose_elitist(pop_current):
    return pop_current[:int(P_size * 0.2)]


def tournament(pop, image):
    shape1 = random.choice(pop)
    shape2 = random.choice(pop)
    fit1 = fitness(image, shape1)
    fit2 = fitness(image, shape2)
    if fit1 < fit2:
        return shape1
    return shape2


def main():
    original_image = plt.imread("image1.png", format="RGB")
    original_image_width = len(original_image)
    original_image_height = len(original_image[0])

    pops_current = []
    num_pops = 2
    for i in range(0, num_pops):
        pops_current.append(init_population(original_image_width, original_image_height))

    for i in range(0, 100):
        print("Step %d" % i)
        for pop_index in range(0, num_pops):
            pop_current = pops_current[pop_index]
            pop_current = sorted(pop_current, key=lambda shape: fitness(original_image, shape),
                                 reverse=False)
            print("DONE FIRST SORT")

            pop_elitist = choose_elitist(pop_current)
            pop_children = []
            children_size = len(pop_current) - len(pop_elitist)

            while len(pop_children) < children_size:
                p1 = tournament(pop_current, original_image)
                p2 = tournament(pop_current, original_image)
                # Cross over
                c1, c2 = cross_over(p1, p2)
                if mutation():
                    mutate(c1, original_image_width, original_image_height)
                if mutation():
                    mutate(c2, original_image_width, original_image_height)
                pop_children.append(c1)
                pop_children.append(c2)

            pop_current = pop_elitist + pop_children
            pops_current[pop_index] = pop_current
        if i % 5 == 0:
            pops_migrating = []
            for pop_index in range(0, num_pops):
                pop_current = pops_current[pop_index]
                pop_current = sorted(pop_current, key=lambda shape: fitness(original_image, shape),
                                     reverse=False)
                pop_migrating = pop_current[:num_pops - 1]
                pops_current[pop_index] = pop_current[num_pops - 1:]
                pops_migrating.append(pop_migrating)

            for pop_index in range(0, num_pops):
                pop_current = pops_current[pop_index]
                pop_arriving = pops_migrating[pop_index - 1]
                pops_current[pop_index] = pop_current + pop_arriving

    for pop_index in range(0, num_pops):
        matrix = create_final_image(pops_current[pop_index], original_image_width, original_image_height)
        img = Image.fromarray(matrix.astype('uint8'))
        img.save("out" + str(pop_index) + ".png")


def mutation():
    return random.random() < 0.001


def create_final_image(shapes, image_width, image_height):
    final_img = np.zeros(shape=(image_width, image_height, 4), dtype='uint8')
    zs = np.zeros(shape=(image_width, image_height))

    for shape in shapes:
        rr, cc = shape.get_coordinates()
        for i in range(0, len(rr)):
            r = rr[i]
            c = cc[i]
            if 0 <= r < shape.image_width and 0 <= c < shape.image_height and shape.z >= zs[r][c]:
                final_img[r][c] = shape.color
                zs[r][c] = shape.z

    return final_img


if __name__ == "__main__":
    main()
