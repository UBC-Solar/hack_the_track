# UBC Solar x Hack the Track
UBC Solar's submission for Hack the Track Presented by Toyota GR

# Quick Start

## For Hackathon Judges
Our deployed app is available at the link provided in our submission.
Simply follow the link to view the app running with Barber Motorsports Park Race 1 data played on a loop

## Running Locally
Docker is required and used to run both the frontend and the backend of the app.

1. Clone the repository

`git clone https://github.com/UBC-Solar/hack_the_track.git`

2. Enter the project root

`cd hack_the_track`

3. Start the docker stack

Linux/MacOS: `sudo docker compose up`
Windows: `docker compose up` (after [installing docker](https://docs.docker.com/desktop/setup/install/windows-install/))

4. Access the web UI at http://localhost:5173/

# Development

## Frontend

To run the React / Vite frontend separately:

1. Create a new terminal and enter the `frontend` directory: `cd frontend`
2. Start the development server: `npm run dev`
3. Access the web UI at http://localhost:5173/

The remainind containers in the docker stack must still be running.


### Neural Network
* Using the provided datasets, we trained a deep recurrent neural network completely from scratch using PyTorch to predict the state evolution of a Toyota GR racecar given the driver’s input to the vehicle. 
* Our model achieved impressive accuracy, being able to correctly predict the position of any car (including cars it wasn’t trained on) within a few meters up to 30 seconds given an initial position and driver input.
* This model is then used to deliver insights on how the driver varying control inputs could improve race performance. Using our industry knowledge of how drivers think, we developed a list of potential variations a driver could undertake at the moment and we report if any variation would result in a significant improvement on race performance.
