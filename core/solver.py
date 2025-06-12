import cv2
import numpy as np
import random
from deap import base, creator, tools, algorithms
from visual import BrushStroke, brushes_to_image, extract_features, load_brush_library, compute_edge_map, convert_rgb_to_lab, unflatten_genome, flatten_genome, compute_gradient_orientation 
from skimage.color import rgb2lab, deltaE_cie76

from skimage.color import deltaE_ciede2000  
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import euclidean_distances

# --- Parameters ---
GENOME_LENGTH = 70
CANVAS_SIZE = (128, 128)

MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.5
ELITE_SIZE = 2
NOVELTY_K = 5


# --- Fitness & Genome Definitions ---
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # fitness ↓, novelty ↑
creator.create("Individual", list, fitness=creator.FitnessMulti)


# --- Novelty Function ---
def compute_novelty(population, brush_library):
    features = [extract_features(brushes_to_image(unflatten_genome(ind), brush_library)) for ind in population]
    distances = euclidean_distances(features)
    novelty_scores = []
    for i in range(len(population)):
        nearest = np.sort(distances[i])[1:NOVELTY_K+1]  # skip self
        novelty = np.mean(nearest)
        novelty_scores.append(novelty)
    return novelty_scores

# --- Edge-based Pixel Sampling ---
def get_edge_filtered_pixels(image_rgb, threshold=50):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=threshold, threshold2=threshold * 3)

    # Get (y, x) coordinates of edge pixels
    edge_coords = np.argwhere(edges > 0)
    edge_pixels = image_rgb[edges > 0]

    return edge_pixels, edge_coords


# --- K-Means Clustering Over Edge Pixels ---
def get_kmeans_clusters(image_rgb, n_clusters=5):
    """
    Uses edge-detected regions to focus clustering on visually important parts of the image.
    Returns cluster_positions as (x, y) coordinates.
    """

    edge_pixels, edge_coords = get_edge_filtered_pixels(image_rgb)

    if len(edge_pixels) < n_clusters:
        raise ValueError("Not enough edge pixels for K-means clustering.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(edge_coords)

    cluster_positions = [(int(x), int(y)) for y, x in kmeans.cluster_centers_]

    return cluster_positions, kmeans.cluster_centers_


# --- Fitness Function ---

def convert_rgb_to_lab(img):
    return rgb2lab(img / 255.0)


def compute_fitness(generated_img, target_lab):
    generated_lab = convert_rgb_to_lab(generated_img)
    diff = deltaE_ciede2000(generated_lab, target_lab)
    return np.mean(diff)


def evaluate_fitness(individual, target_lab, brush_lib):
    brushes = unflatten_genome(individual)
    rendered = brushes_to_image(brushes, brush_lib, canvas_size=CANVAS_SIZE)
    
    individual.rendered = rendered  # <-- 🧠 store rendered image on the individual
    
    fitness = compute_fitness(rendered, target_lab)
    return fitness, rendered


# ---Evolution setup ---

def run_solver(target_rgb, gradient_map, generations=50, pop_size=40):
    brush_library = load_brush_library("brushes/watercolor")
    brush_count = len(brush_library)
    print(f"[📦] Total brushes loaded: {brush_count}")
    target_lab = convert_rgb_to_lab(target_rgb)
    gradient_map = compute_gradient_orientation(target_rgb)

    # gradient_map is a normalized edge magnitude image (0-255)
    edges = compute_edge_map(target_rgb)

    def evolve_stage(prev_best=None, genome_length=20, max_size=0.15, generations=10):

        def create_individual():
            cluster_positions, cluster_colors = get_kmeans_clusters(target_rgb, n_clusters=10)
            brushes = []

            edge_probs = edges.flatten().astype(np.float64)

            # Clip negatives and add small epsilon to avoid division by zero
            edge_probs = np.clip(edge_probs, 0, None)
            total = edge_probs.sum()
            if total == 0:
                edge_probs[:] = 1  # fallback to uniform sampling
            else:
                edge_probs /= total

            for _ in range(genome_length):

                idx = random.randint(0, len(cluster_positions) - 1)
                x, y = cluster_positions[idx]
                color = tuple(cluster_colors[idx])
                angle = gradient_map[y, x] + random.uniform(-30, 30)
                brushes.append(BrushStroke(
                    brush_id=random.randint(0, brush_count - 1),
                    x=x,
                    y=y,
                    size=random.uniform(0.05, max_size),
                    rotation=angle,
                    color=color,
                    opacity=random.uniform(0.3, 0.8)
                ))
            return creator.Individual(flatten_genome(brushes))

        def evaluate(ind):
            fitness_score, _ = evaluate_fitness(ind, target_lab, brush_library)
            return fitness_score, 0.0  # Dummy novelty

        def mutate(ind):
            i = random.randrange(len(ind))
            if i % 9 == 0:
                ind[i] = int(random.randint(0, brush_count - 1))
            elif i % 9 in [1, 2]:
                ind[i] = int(np.clip(ind[i] + random.randint(-10, 10), 0, CANVAS_SIZE[1 if i % 9 == 1 else 0] - 1))
            elif i % 9 == 3:
                ind[i] = np.clip(ind[i] + random.uniform(-0.05, 0.05), 0.01, max_size)
            elif i % 9 == 4:
                ind[i] = (ind[i] + random.uniform(-15, 15)) % 360
            elif i % 9 in [5, 6, 7]:
                ind[i] = int(np.clip(ind[i] + random.randint(-20, 20), 0, 255))
            elif i % 9 == 8:
                ind[i] = np.clip(ind[i] + random.uniform(-0.1, 0.1), 0.0, 1.0)
            return ind,

        def crossover(ind1, ind2):
            for i in range(len(ind1)):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2

        toolbox = base.Toolbox()
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=pop_size)

        if prev_best:
            for i in range(len(population)):
                population[i][:len(prev_best)] = prev_best[:]

        fitness_log = []

        for gen in range(generations):
            # Evaluate base pop
            for ind in population:
                fit, _ = evaluate_fitness(ind, target_lab, brush_library)
                ind.fitness.values = (fit, 0.0)

            novelty_scores = compute_novelty(population, brush_library)
            for i, ind in enumerate(population):
                ind.fitness.values = (ind.fitness.values[0], novelty_scores[i])

            population = tools.selNSGA2(population, len(population))
            offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)

            for ind in offspring:
                fit, _ = evaluate_fitness(ind, target_lab, brush_library)
                ind.fitness.values = (fit, 0.0)
            novelty_scores = compute_novelty(offspring, brush_library)
            for i, ind in enumerate(offspring):
                ind.fitness.values = (ind.fitness.values[0], novelty_scores[i])

            population = tools.selNSGA2(population + offspring, pop_size)

            best = tools.selBest(population, 1)[0]
            print(f"Gen {gen+1} | Fitness: {best.fitness.values[0]:.4f} | Novelty: {best.fitness.values[1]:.4f}")
            fitness_log.append(best.fitness.values[0])

        best_final = tools.selBest(population, 1)[0]
        return population, fitness_log, best_final

    # --- Coarse-to-Fine Evolution Stages ---

    stages = [
    {"length": 40, "size": 0.15, "gens": 20},
    {"length": 80, "size": 0.08, "gens": 30},
    {"length": 150, "size": 0.05, "gens": 50},
]

    final = None
    logs = []
    for stage in stages:
        print(f"\n🧪 Stage: {stage['length']} strokes, {stage['gens']} generations")
        pop, log, best = evolve_stage(
            prev_best=final,
            genome_length=stage["length"],
            max_size=stage["size"],
            generations=stage["gens"]
        )
        final = best
        logs.extend(log)

    return pop, logs, brush_library
