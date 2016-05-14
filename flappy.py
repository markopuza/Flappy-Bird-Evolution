import pygame
from pygame.locals import *
from neat import nn, population

from itertools import cycle
import random as rnd
import time
import math

#####################################
# Game parameters
#####################################

FPS, WIDTH, HEIGHT = 30, 288, 512
SPACING = WIDTH // 2 # distance between pipes
GAME_VEL = -4 # velocity of the game

generation, highscore = 0, 0 # statistic counters
BASEX, BASEY = 0, int(0.79 * HEIGHT) # position of ground
GAP = 100 # gap in the pipes

path = 'assets/'
IMAGES, HITMASKS = {}, {}

#####################################
# Bird class
#####################################

class Bird:
    ''' Contains all data about the bird and his neural network '''
    def __init__(self, genome):
        self.genome = genome
        self.color = ['blue', 'red', 'yellow', 'black'][rnd.randint(0,3)]
        self.state = 0 # denotes the state of the wing
        self.generator = cycle([0,1,2,1]) # iterator of the wing states
        self.alive = True
        self.jumps = 0 # counts total number of flaps

        self.x = WIDTH // 5
        self.y = HEIGHT // 2.5

        self.velocity = -8 # vertical velocity
        self.acceleration = 1 # vertical acceleration

        self.brain = nn.create_feed_forward_phenotype(genome)

    def image(self):
        ''' called at each tick of the clock '''
        self.state = self.generator.next() # move the wing
        self.y = min(self. y + self.velocity, BASEY - 23) # update position
        if self.velocity < 12: # update velocity
            self.velocity += self.acceleration
        return IMAGES[self.color+'-'+str(self.state)]

    def flap(self):
        if self.y < 10: # bird is too high to flap
            return
        self.jumps += 1
        if self.velocity < 0: # midflap flap
            self.velocity = max(-12, self.velocity - 8)
        else: # non-midflap flap
            self.velocity = -8

    def decision(self, pipes):
        ''' decides whether to flap or not '''
        # positions of the two leftmost pipes on the screen
        p1x, p1y = pipes[0][1][0], pipes[0][1][1]
        p2x, p2y = pipes[1][1][0], pipes[1][1][1]

        # set normalized velocity as one of the inputs
        inputs = [1.0, float(self.velocity + 13) / 25]

        # bird only sees the pipe that is IN FRONT of him,
        # so add the normalized relative position of the pipe to inputs
        if p1x - self.x >= -40:
            inputs += [float(p1x) / WIDTH, (float(p1y) - self.y) / HEIGHT]
        else:
            inputs += [float(p2x) / WIDTH, (float(p2y) - self.y) / HEIGHT]

        # get output of the neural network
        output = self.brain.serial_activate(inputs)
        # and decide
        if output[0] > 0.5:
            self.flap()

    def collided(self, pipes):
        ''' finds out whether the bird is in collision right now '''
        if self.y + 24 >= BASEY: # with ground
            return True

        bird_rect = pygame.Rect(self.x, self.y, 34, 24)
        for pipe in pipes: # with pipes
            pipe_up_rect = pygame.Rect(pipe[0][0], pipe[0][1], 52, 320)
            pipe_down_rect = pygame.Rect(pipe[1][0], pipe[1][1], 52, 320)

            if pixelCollision(bird_rect, pipe_up_rect, HITMASKS['bird-'+str(self.state)], HITMASKS['pipe_up']):
                return True
            if pixelCollision(bird_rect, pipe_down_rect, HITMASKS['bird-'+str(self.state)], HITMASKS['pipe_down']):
                return True
        return False

#####################################
# Helper methods
#####################################

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_mask(image):
    ''' returns pixels with zero alpha channel '''
    mask = []
    for i in range(image.get_width()):
        mask.append([])
        for j in range(image.get_height()):
            mask[i].append(bool(image.get_at((i,j))[3]))
    return mask

def random_pipe(x):
    ''' returns coordinates of upper and lower pipes at position x '''
    gapY = rnd.randrange(0, int(BASEY * 0.6 - GAP)) + BASEY // 5
    pipe_height = IMAGES['pipe_down'].get_height()
    return([(x, gapY - pipe_height),(x, gapY + GAP)])

def show_score(score):
    ''' displays score in center of screen '''
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0
    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()
    Xoffset = (WIDTH - totalWidth) / 2
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, HEIGHT // 10))
        Xoffset += IMAGES['numbers'][digit].get_width()

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide """
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

#####################################
# Game loop for one generation
#####################################

def eval_fitness(genomes):
    global BASEX, generation, highscore

    birds = []
    for genome in genomes:
        birds.append(Bird(genome)) # a bird is born

    birds_alive = len(birds)
    for i in range(len(birds)):
        birds[i].y += 5 * rnd.randint(-5,5)

    # initialize pipe array
    pipes = [random_pipe(WIDTH + 100 + i * SPACING) for i in range(3)]

    # initialize score and time
    score, score_added = 0, False
    start = time.time()

    # play until there is a survivor
    while birds_alive:
        SCREEN.blit(IMAGES['background'], (0, 0))
        for p in pipes:
            SCREEN.blit(IMAGES['pipe_up'], p[0])
            SCREEN.blit(IMAGES['pipe_down'], p[1])
        SCREEN.blit(IMAGES['base'], (BASEX, BASEY))

        # move everything that needs movement
        BASEX = -((-BASEX + 4) % 48)
        for p in pipes:
            p[0] = (p[0][0] + GAME_VEL, p[0][1])
            p[1] = (p[1][0] + GAME_VEL, p[1][1])
        if not score_added and pipes[0][0][0] < WIDTH // 5:
            score += 1
            score_added = True
        if pipes[0][0][0] < -50: # get rid of the left pipe and add a new one
            pipes = pipes[1:]
            pipes.append(random_pipe(pipes[1][0][0] + SPACING))
            score_added = False

        for b in birds:
            if not b.alive:
                continue
            if b.collided(pipes):
                b.alive = False
                birds_alive -= 1
                highscore = max(highscore, score)
                lifespan = float(time.time() - start)
                # FITNESS FUNCTION
                # takes into account score, # of flaps and lifespan
                b.genome.fitness = sigmoid(score / 10.0 - b.jumps / 1000.0 + lifespan / 200.0)
            b.decision(pipes)
            SCREEN.blit(b.image(), (b.x, b.y))

        # print statistics
        label1 = FONT.render('Alive: ' + str(birds_alive), 2, (0,0,0))
        label2 = FONT.render('HIGHSCORE: ' + str(highscore), 2, (0,0,0))
        label3 = FONT.render('GENERATION: ' + str(generation), 2, (0,0,0))
        SCREEN.blit(label1, (20, 440))
        SCREEN.blit(label2, (20, 460))
        SCREEN.blit(label3, (20, 480))
        show_score(score)

        # update the screen and tick the clock
        pygame.display.update()
        FPSCLOCK.tick(FPS)
    generation += 1

#####################################
# Main skeleton
#####################################

def main():
    # initalize game
    global FPSCLOCK, SCREEN, FONT
    pygame.init()
    FONT = pygame.font.SysFont("monospace", 15)
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Flappy Bird Evolution')

    # get game element images
    day_time = ['day', 'night'][rnd.randint(0,1)]
    IMAGES['background'] = pygame.image.load(path+'background-'+day_time+'.png').convert()
    IMAGES['base'] = pygame.image.load(path+'base.png').convert()
    IMAGES['message'] = pygame.image.load(path+'message.png').convert_alpha()
    IMAGES['pipe_down'] = pygame.image.load(path+'pipe-red.png').convert_alpha()
    IMAGES['pipe_up'] = pygame.transform.rotate(IMAGES['pipe_down'], 180)
    IMAGES['numbers'] = [pygame.image.load(path + str(i) + '.png').convert_alpha() for i in range(10)]
    for color in ['blue', 'red', 'yellow', 'black']:
        for state in range(3):
            IMAGES[color+'-'+str(state)] = pygame.image.load(path+color+'bird-'+str(state)+'.png').convert_alpha()

    # get hitmasks
    for i in range(3):
        HITMASKS['bird-'+str(i)] = get_mask(pygame.image.load(path+'bluebird-'+str(i)+'.png').convert_alpha())
    HITMASKS['pipe_up'] = get_mask(IMAGES['pipe_up'])
    HITMASKS['pipe_down'] = get_mask(IMAGES['pipe_down'])

    # show welcome screen
    SCREEN.blit(IMAGES['background'], (0, 0))
    SCREEN.blit(IMAGES['message'], (45, 45))
    SCREEN.blit(IMAGES['base'], (BASEX, BASEY))
    pygame.display.update()

    # hold until space is pushed
    hold = True
    while hold:
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_SPACE:
                hold = False

    # start evolution
    pop = population.Population('flappy_config')
    pop.run(eval_fitness, 10000)

if __name__ == '__main__':
    main()
