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

# Submission Details

## Inspiration
Our greatest inspiration for participating and for our submission is how close to home this hackathon hits. Reading the intro of the hackathon fired us up because this is exactly the type of work we aim to do in the Solar Car Racing community. So, getting a chance to apply it elsewhere, potentially collaborate with TRD, improve our own systems, and, of course, win some prizes is exactly why we joined!

Our submissions are inspired by our observations of where we believe the TRD community will benefit the most. Seeing that in this specific competition the car specs are the same but the **driver** as the variable immediately triggered real-time driver insights so we were motivated to completely hone in on this aspect for the greatest impact. The reason why is because maximizing team success means accelerating the areas with the most potential, and in this case that's the drivers!

## What it does
This application helps train drivers by replaying previous races or watching their races live and seeing various insights about how to improve race times. Through the UI, Race Engineers can choose certain cars on the track and see specific insights for that car such as increasing or decreasing braking or steering angle. Currently this application works with the Barber track with GPS data, however, if any race with GPS data is provided then there is lots of room to learn from those races! We generate driver insights every 5 seconds because, at high speeds, it provides sufficient distance for meaningful adjustments to braking and steering without overwhelming the driver. Additionally, the driver insights list provides Race Engineers with a view of all the improvements needed so a cohesive training plan can be made to target the most common improvements (Ex. Braking 25% less into specific turns). 

### Specifics
* When you select the vehicle it will be a **red** dot with a **black** outline. All the lap times and current lap data will change depending on the vehicle selected. For example, choose "GR88-022-13".
* Hover over other dots to see what car it is.
* Enable or disable 'Data Collection' to effectively pause the simulation. This give you time to think about driver insights! Time is still ticking so unpausing will update the cars to their latest location instead of simply continuing off. We did this to make our application real-time.  
* Lap count and driver insights are automatically generated!

![Our application's UI. Cars shown as dots on the Barber Motorsports Park Track, with floating UI elements and driver insights.](https://i.imgur.com/zKEysnB.jpeg)

## Why Our Submission Shines
* Instead of re-formatting the existing data (Ex. with calculations, heuristics or indexes) to inform the engineers of the driver's or car's **state**, we aimed to do what a human cannot; compute many possibilities instantly and find the best one. And furthermore, find the **exact** time gain and action the driver should have taken. This gives the most actionable training and feedback!
* **Research**: We spent days clearing up requirements for our project to make absolutely sure we designed a tool with the greatest impact. This meant learning what TRD community already uses, justifying every single decision with papers or analysis, and defining a layout that is intuitive to Race Engineers. For instance, there are many free AI Analytic Telemetry UI's available for this use case (already on GitHub and the majority of submissions from this hackathon). Knowing this, we focused more effort to produce the most actionable and accurate results that teams will use. Future improvements include adding very specific components **not** available at all.  
* Our deep RNN architecture delivers **precise, quantitative adjustments** rather than vague suggestions like “brake later” or “add steering.” It also computes the expected time gain with high accuracy. To validate this, we compared our system against general-purpose AIs such as Gemini and ChatGPT. When we fed each model identical inputs, exact changes to braking strength, steering angle, and more, and asked for 100 predicted positions at 50 ms intervals over 5 seconds, we plotted the resulting **predicted racing lines**. Our model consistently produced the most physically realistic trajectories. These exact, actionable corrections are critical for driver development because they tell racers **exactly how much** to adjust. Ultimately, we set out to build a serious, practical tool, one that the TRD community, our Solar Car Racing team, and many others will benefit from.

## How we built it
* Raw CSV data from any race is uploaded to our Telemetry Database using PostgreSQL.
* Then our FastAPI backend queries this data based on commands from our front end such as the car number and the current time. This way we simulate past races in real-time or current races **live**. Kafka is used for the data streaming here. *We learned that large telemetry datasets are best managed when streamed instead of bulk loading!*. 
* We use PyTorch for performing inference on the car state and controls such as acceleration, steering, and braking. We use this to generate possible racing lines and then find the optimal one.
* Then our React front end will display the driver insights and the race!
* AWS EC2 instance for hosting and Cloudflare to use a simple URL that anyone can remember and instantly access!

![Logos of the tools used in our stack](https://i.imgur.com/NKdhxQz.png)

![System diagram of our app](https://i.imgur.com/a7XWqS8.png)

### Control Modifications
From a [study comparing humans to machines in racing](https://orca.cardiff.ac.uk/id/eprint/174114/1/On_the_human-machine_gap_in_car_racing_A_comparative_analysis_of_machine_performance_against_human_drivers.pdf), we found driver input such as gear changes only occur a handful of times in contrast to micro-corrections in steering angle and acceleration that compound into real time-loss. Thus, we designed our tool to target the most prominent controls instead of just trying them all (reducing real-time capability on 90% of devices).

Additionally, our simple architecture leaves room for Engineers to add their own control modifications targeted for their specific driver. For example, consider adding:
* More aggressive steering angle corrections
* Gear up and down correction

### Neural Network
* Using the provided datasets, we trained a deep recurrent neural network completely from scratch using PyTorch to predict the state evolution of a Toyota GR racecar given the driver’s input to the vehicle. 
* Our model achieved impressive accuracy, being able to correctly predict the position of any car (including cars it wasn’t trained on) within a few meters up to 30 seconds given an initial position and driver input.
* This model is then used to deliver insights on how the driver varying control inputs could improve race performance. Using our industry knowledge of how drivers think, we developed a list of potential variations a driver could undertake at the moment and we report if any variation would result in a significant improvement on race performance.

![Neural network result, showing multiple simulated racing line compared to the actual path.](https://i.imgur.com/btHGYxe.png)

## Challenges we ran into
* **The Neural Network**: This is by far the biggest challenge we faced. Honestly, we were not sure if we could achieve sufficient accuracy with our model before the deadline. Throughout the hackathon we tried different methods such as modelling the cars via physics equations, however, these previous attempts were unsuccessful. Feeling discouraged, we turned to thinking more practically to avoid a sunken cost fallacy where we only worked on the Neural Network. We divided up work to prioritize filming our submission and polishing what we had while others chewed away at the NN. All the long hours spent making tweaks all over the project to ensure it works to our standard were well worth it to produce a practical application for real Race Engineers. 

## Accomplishments that we're proud of
* **Also the Neural Network 🤖**: Despite all the long nights and hardship, our strategy team pulled through and fleshed out a Neural Network and an optimizer to generate driver insights. 
* **Our Submission Video 📹**: We truly wanted to demonstrate the usefulness of our application through examples of a TRD Race Team's workflow when using our application. We spent time directing, shooting, and re-shooting shots to make sure the message we intended was delivered. In general, to produce an excellent demonstration, we aimed to put ourselves in the shoes of the Race Engineers. Furthermore, we considered our own workflow when designing our strategy at competition, which helped us brainstorm various shots to film! 
* **Our persistence 💪**: Overall, for both the points above, it was our continual effort to design and provide a tool for the TRD community and also for many communities beyond that we are proud of. Specifically, ensuring the end result will be useful is something we honed in on. Whether it was researching papers that indicate our is impactful or learning a lot about Toyota's GR Program, we were persistent to develop a practical, lasting, and central tool.

## What we learned
Overall, **a lot**! Below are just the highlights but in reality we spent hours and hours trying ideas and debugging until we met our requirements. 
* **Using a Neural Network**: We do not use a NN for our Race Strategy back at Solar. Instead we model various parameters of our car with physics equations. However, seeing the potential for driver insights through the use of a NN acting on telemetry data, we are excited to explore how we can integrate the NN into our existing pipeline.
* **Stack Development**: It was awesome to choose various applications for our stack, learn about each of them, and deliver a practical project that uses them. Not all members of our team are familiar with these applications so pushing our team to the same playing field served for a great learning opportunity.
* **Project Management**: When you are close to the deadline the previous mistakes and inefficiencies hit like a truck. We reflected and learned how to structure our work more effectively, what information should always be communicated (progress and documentation), and how to develop a successful plan.

![UBC Solar members working hard on Hack The Track](https://i.imgur.com/qSyjfe6.jpeg)

## What's next for UBC Solar's Driver Insights
* **Importing new Tracks**:  Currently, the provided dataset only includes GPS for one race. However, we want our tool to provide insight for *any* race to provide continuous driver growth. So, we will build tools allowing users to work with new track maps for their region or series.
* **Delivering a Driver-Insights API**: Instead of analyzing each race individually, why not scale to entire seasons and uncover deeper driver behavior patterns. This helps drivers not only improve their skills even more but see their hard work pay off.

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
