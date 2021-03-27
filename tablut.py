import datetime
import os

import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (5, 9, 9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(6561))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 225  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward (1 for board games)
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 100  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

#Conversione delle coordinate in numero:
#(l,k)->(j,i) Rappresenta la mossa da effettuare nella board 9x9
#Per convertire in numero: l*9^3+k*9^2+j*9^1+i*9^0
#Per convertire in coordinate: Operazione di divisione e resti

#Esempio
#(1,0)->(0,0) => 1*9^3+0*9^2+0*9^1+0*9^0=729
#729 // 9 = 81 (729 % 9 = 0 (i))
#81 // 9 = 9 (81 % 9 = 0 (j))
#9 // 9 = 1 (9 % 9 = 0 (k))
#1 // 9 = 0 (1 % 9 = 1 (l))

def coords_to_number(coords):
    l,k = coords[0]
    j,i = coords[0]
    return l*729 + k*81 + j*9 + i

def number_to_coords(number):
    i = number % 9
    number = number // 9
    j = number % 9
    number = number // 9
    k = number % 9
    number = number // 9
    l = number % 9
    number = number // 9

    return ((l,k),(j,i))

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = AshtonTablut()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                l = int(
                    input(
                        f"Enter the start row (1,..,9) to play for the player {self.to_play()}: "
                    )
                )
                k = int(
                    input(
                        f"Enter the start column (1,..,9) to play for the player {self.to_play()}: "
                    )
                )
                j = int(
                    input(
                        f"Enter the end row (1,..,9) to play for the player {self.to_play()}: "
                    )
                )
                i = int(
                    input(
                        f"Enter the end column (1,..,9) to play for the player {self.to_play()}: "
                    )
                )
                choice = coords_to_number(tuple((l-1,k-1), (j-1,i-1)))
                if (
                    choice in self.legal_actions()
                    and 1 <= l
                    and 1 <= k
                    and l <= 9
                    and k <= 9
                    and 1 <= j
                    and 1 <= i
                    and j <= 9
                    and i <= 9
                ):
                    break
            except:
                pass
            print("Wrong input, try again")

        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        coords = number_to_coords(action_number)
        return f"{coords[0]} -> {coords[1]}"


class AshtonTablut:
    def __init__(self):
        self.board = np.zeros((3, 9, 9), dtype=np.int8)
        self.player = -1 #-1: Bianco, 1: Nero
        self.drawQueue = []
        self.stepsWithoutCapturing = 0

    def to_play(self):#Players: [0,1]: 0 is White, 1 is Black
        return 0 if self.player == -1 else 1

    def reset(self):
        self.player = -1
        
        self.board = np.zeros((4, 9, 9), dtype=np.int8)
        
        #Board[0]: Bianco altro 0
        self.board[0] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

        #Board[1]: Nero altro 0
        self.board[1] = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0], 
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 0, 0, 0, 0, 0, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.int8)

        #Board[2]: Re 1 altro 0
        self.board[2] = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
        
        #Board[3]: Celle non calpestabili: citadels, trono 1 calpestabili 0
        #Rimozione di alcune citadels ((0,4), (4,0), (4,8), (8,4)): per evitare che il nero sia mangiato quando dentro alla citadels

        self.board[3] = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 1, 0, 0, 0]], dtype=np.int8)

        return self.get_observation()

    def step(self, action):
        action = number_to_coords(action)
        playerBoard = self.board[self.to_play()]

        fromYX = action[0]
        toYX = action[1]

        tmp = playerBoard[fromYX]
        playerBoard[fromYX] = 0
        playerBoard[toYX] = tmp

        eaten = 0

        #Controllo se mangio pedine
        if self.to_play() == 0:
            eaten = self.check_white_eat(action)
        else:
            eaten = self.check_black_eat(action)

        #Controllo se non ho mangiato pedine
        if eaten > 0:
            self.stepsWithoutCapturing = 0
            self.drawQueue = []
        else:
            self.stepsWithoutCapturing += 1
            self.drawQueue.append(self.board[:3].copy())

        self.player *= -1

        winCheck = self.have_winner() or len(self.legal_actions()) == 0
        drawCheck = self.have_draw()
        done = winCheck or drawCheck

        reward = 1 if winCheck else 0

        return self.get_observation(), reward, done

    def get_observation(self):
        board_to_play = np.full((9, 9), self.player, dtype=np.int8)
        return np.array([self.board[0], self.board[1], self.board[2], self.board[3], board_to_play], dtype=np.int8)

    def legal_actions(self):
        legal = []

        #Creo una maschera: pedoni, re, cittadelle
        mask = self.board[0] | self.board[1] | self.board[2] | self.board[3]
        
        #Seleziono i pedoni del giocatore
        pedoni = np.where(self.board[self.to_play()] == 1)

        for y,x in zip(pedoni[0], pedoni[1]):
            #Seleziono le celle adiacenti (no diagonali)
            #Appena incontro un'ostacolo mi fermo (no salti, nemmeno il trono)

            #Su
            for newY in reversed(range(y)):
                if mask[newY,x] == 0:
                    legal.append(coords_to_number(((y,x), (newY,x))))
                else:
                    break

            #Giu
            for newY in range(y,9):
                if mask[newY,x] == 0:
                    legal.append(coords_to_number(((y,x), (newY,x))))
                else:
                    break

            #Sinistra
            for newX in reversed(range(x)):
                if mask[y,newX] == 0:
                    legal.append(coords_to_number(((y,x), (y,newX))))
                else:
                    break

            #Destra
            for newX in range(x,9):
                if mask[y,newX] == 0:
                    legal.append(coords_to_number(((y,x), (y,newX))))
                else:
                    break

        return legal

    def check_black_eat(self, action):#Controllo se il nero mangia dei pedoni bianchi
        y,x = action[1]#Dove è finita la pedina nera che dovrà catturare uno o più pedoni bianchi?
        captured = 0

        #Le citadels possono fare da spalla
        allies = self.board[1] | self.board[3]
        enemies = self.board[0]

        #Seleziono le quattro terne di controllo
        lookUp = np.array([allies[y-2:y+1,x], enemies[y-2:y+1,x]])
        lookDown = np.array([allies[y:y+3,x], enemies[y:y+3,x]])
        lookLeft = np.array([allies[y,x-2:x+1], enemies[y,x-2:x+1]])
        lookRight = np.array([allies[y,x:x+3], enemies[y,x:x+3]])

        captureCheck = np.array([[1,0,1], [0,1,0]])

        if np.array_equal(lookUp, captureCheck):
            self.board[1, y-1, x] = 0
            captured +=1
        if np.array_equal(lookDown, captureCheck):
            self.board[1, y+1, x] = 0
            captured +=1
        if np.array_equal(lookLeft, captureCheck):
            self.board[1, y, x-1] = 0
            captured +=1
        if np.array_equal(lookRight, captureCheck):
            self.board[1, y, x+1] = 0
            captured +=1

        return captured

    def check_white_eat(self, action):#Controllo se il bianco mangia dei pedoni neri
        y,x = action[1]#Dove è finita la pedina bianca che dovrà catturare uno o più pedoni neri?
        captured = 0

        #Il re può fare da spalla
        #Le citadels possono fare da spalla
        allies = self.board[0] | self.board[2] | self.board[3]
        enemies = self.board[1]

        #Seleziono le quattro terne di controllo
        lookUp = np.array([allies[y-2:y+1,x], enemies[y-2:y+1,x]])
        lookDown = np.array([allies[y:y+3,x], enemies[y:y+3,x]])
        lookLeft = np.array([allies[y,x-2:x+1], enemies[y,x-2:x+1]])
        lookRight = np.array([allies[y,x:x+3], enemies[y,x:x+3]])

        captureCheck = np.array([[1,0,1], [0,1,0]])

        if np.array_equal(lookUp, captureCheck):
            self.board[1, y-1, x] = 0
            captured +=1
        if np.array_equal(lookDown, captureCheck):
            self.board[1, y+1, x] = 0
            captured +=1
        if np.array_equal(lookLeft, captureCheck):
            self.board[1, y, x-1] = 0
            captured +=1
        if np.array_equal(lookRight, captureCheck):
            self.board[1, y, x+1] = 0
            captured +=1

        return captured

    def have_draw(self):
        #Controllo se ho un certo numero di stati ripetuti
        trovati = 0
        for state in self.drawQueue:
            if np.array_equal(state, self.board[:3]):
                trovati +=1
        
        if trovati > 0:
            return True

        return False

    def have_winner(self):
        #White Check
        if self.white_win_check():
            return True

        #Black Check
        if self.black_win_check():
            return True

        return False

    def white_win_check(self):
        #Controllo che il Re sia in un bordo della board
        top = np.sum(self.board[2, 0])
        down = np.sum(self.board[2, 8])
        left = np.sum(self.board[2, :, 0])
        right = np.sum(self.board[2, :, 8])

        return top == 1 or down == 1 or left == 1 or right == 1

    def black_win_check(self):
        #Controllo se il nero ha catturato il re

        #Se il re è sul trono allora 4
        #Se il re è adiacente al trono allora 3 pedoni che lo circondano
        #Altrimenti catturo come pedone normale (citadels possono fare da nemico)

        king = np.where(self.board[2] == 1)
        
        if king == (4,4):#Re sul trono. Controllo i bordi (3,4), (4,3), (4,5), (5,4)
            if self.board[1, 3, 4] == 1 and self.board[1, 4, 3] == 1 and self.board[1, 4, 5] == 1 and self.board[1, 5, 4] == 1:
                return True
        
        elif king in ((3,4), (4,3), (4,5), (5,4)):#Re adiacente al trono: controllo se sono presenti nemici intorno
            #Aggiungo il trono alle pedine nemiche (in realtà aggiungo anche le citadels ma non influenzano)
            enemies = self.board[1] | self.board[3]
            y,x = king
            if enemies[y-1, x] == 1 and enemies[y+1, x] == 1 and enemies[y, x-1] == 1 and enemies[y, x+1] == 1:
                return True

        else:#Check cattura normale.
            #Aggiungo i contraints
            enemies = self.board[1] | self.board[3]
            y,x = king
            if enemies[y-1, x] == 1 and enemies[y+1, x] == 1 or enemies[y, x-1] == 1 and enemies[y, x+1] == 1:
                return True

        return False

    def render(self):
        print(-self.board[0]+self.board[1]-20*self.board[2]+self.board[3])
