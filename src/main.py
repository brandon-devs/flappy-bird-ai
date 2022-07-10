# Importing modules
import pygame
import os
import random
import neat
import math

# Setting up the window
WIDTH = 1000
HEIGHT = 750
FPS = 60

pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird AI")
clock = pygame.time.Clock()

# Bird class
class Bird:
  def __init__(self):
    self.image = pygame.image.load(os.path.join("Assets", "bird.png"))
    self.bird = pygame.transform.scale(self.image, (50, 50))
    self.x = 100
    self.y = 200

# Tube class
class Tube:
    def __init__(self):
        self.num = random.randint(350, WIDTH - 350)
        self.top = pygame.Rect(WIDTH, 0, 50, self.num - 100)
        self.bottom = pygame.Rect(WIDTH, self.num + 100, 50, WIDTH - self.num + 100)  

# Creating Pygame events
move_tube_time = 100 
move_tube_event = pygame.USEREVENT + 1
pygame.time.set_timer(move_tube_event, move_tube_time)

create_tube_time = 2000
create_tube_event = pygame.USEREVENT + 2
pygame.time.set_timer(create_tube_event, create_tube_time)

# Main loop that NEAT will run
def eval_genomes(genomes, config):

  birds = []
  tubes = []

  # Setting up the genomes and neural networks
  genomes_list = []
  nets = []
  
  # Function to completely remove a "player"
  def destroy(index):
    genomes_list.pop(index)
    nets.pop(index)
    birds.pop(index)

  for genome_id, genome in genomes:
    genomes_list.append(genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nets.append(net)
    genome.fitness = 0
    birds.append(Bird())

  # Constant game loop
  on = True
  while on:

      # Controlling the speed of the game
      clock.tick(FPS) 

      # Checking for Pygame events to create and move tubes
      for event in pygame.event.get():
          if event.type == move_tube_event:
              for tube in tubes:
                  tube.top.x -= 10
                  tube.bottom.x -= 10
          if event.type == create_tube_event:
              tubes.append(Tube())
          if event.type == pygame.QUIT:
              on = False

      # Drawing the birds and tubes onto the screen
      window.fill((255, 255, 255))

      for bird in birds:
          window.blit(bird.bird, (bird.x, bird.y))

      for tube in tubes:
          pygame.draw.rect(window, (0, 255, 0), tube.top)
          pygame.draw.rect(window, (0, 255, 0), tube.bottom)

      # Detecting collisions
      for bird in birds:
        if bird.y > HEIGHT - 25 or bird.y < 0:
          destroy(birds.index(bird))
  
      for tube in tubes:
        for bird in birds:
          bird_rect = bird.bird.get_rect(center=(bird.x + 25, bird.y + 25))
          if tube.top.colliderect(bird_rect):
            destroy(birds.index(bird))
          if tube.bottom.colliderect(bird_rect):
            destroy(birds.index(bird))

      # Checks if the birds passed through a tube - if so, adds fitness
      for tube in tubes:
        for bird in birds:
          if tube.top.x - 10 < bird.x:
            genomes_list[birds.index(bird)].fitness += 10

      # Increasing the fitness of remaining birds
      for bird in birds:
        genomes_list[birds.index(bird)].fitness += 0.1

      # Activating the genome and sending a request to the nueral network
      for bird in birds:
        if tubes != []:
          distances = []
          for tube in tubes:
            dist = math.sqrt((bird.x - tube.top.x)**2 + (bird.y - tube.top.y)**2)
            distances.append(dist)
          closest_tube = tubes[distances.index(min(distances))]
          output = nets[birds.index(bird)].activate((float(bird.y), float(closest_tube.top.x), float(closest_tube.top.y), float(closest_tube.bottom.x), float(closest_tube.bottom.y)))
          if output[0] > 0.5:
            bird.y -= 25
          else:
            bird.y += 5

      if birds == []:
        return
    
      pygame.display.update()
  
  pygame.quit()

# NEAT AI setup
def run(config_file):
  config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
  population = neat.Population(config)
  population.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  population.add_reporter(stats)
  population.add_reporter(neat.Checkpointer(5))
  population.run(eval_genomes, 25)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
