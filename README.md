# Nexa Visualize

Nexa Visualize is an interactive,
educational tool for visualizing the architecture and training process of a neural network in real-time 3D.
Built with pure JavaScript and Three.js,
it provides an intuitive way to understand complex concepts like forward propagation,
backpropagation, and the impact of hyperparameter tuning.

## Features

-   **Interactive 3D Visualization:** View neural network architectures in a dynamic 3D space.
-   **Dynamic Architecture:** Add or remove hidden layers and change the number of neurons in real-time to see how the structure changes.
-   **Real-time Training Animation:** Observe the training process with clear, distinct animations:
    -   **Forward Pass:** A green wave of light flows from the input to the output layer.
    -   **Backward Pass:** A red wave flows in reverse, simulating gradient calculation.
    -   **Weight Update:** A blue "pop" effect on neurons signifies that the weights have been updated.
-   **Customizable Hyperparameters:** Adjust parameters like epochs, learning rate, optimizer, and loss function to see their immediate impact on training.
-   **Live Performance Monitoring:** Track training progress with a real-time graph that plots training loss (red) and accuracy (green).
-   **Interactive Camera:** Freely explore the 3D scene with an autorotation mode or manual drag-to-rotate and scroll-to-zoom controls.
-   **Clean, Non-Distracting UI:** A simple, expandable control panel keeps the focus on the visualization, showing customization options only when you need them.

## Controls

### Main Controls
-   **Start Training / Stop:** Begins or halts the training simulation.
-   **Reset:** Resets the training progress and network state to its initial configuration.
-   **Auto / Manual:** Toggles between automatic camera rotation and manual user control.
-   **Speed Slider:** Adjust the speed of the training animation.

### Customization Panels
Click a category button to expand its details. Click it again to collapse.
-   **Architecture:** Add or remove hidden layers and set the number of neurons for each layer.
-   **Parameters:** Adjust training hyperparameters like Epochs, Learning Rate, Optimizer, and Loss Function.
-   **Monitoring:** View the live graph of training loss and accuracy, along with key parameter info.

## How to Run

To view this project, you need to run it from a local web server. You cannot simply open `public/index.html` as a file in your browser due to security policies (CORS).

Here are a few easy ways to start a local server:

### Option 1: Using Docker (Recommended)

You can build and run this project in a Docker container for easy deployment on any OS.

1. **Build the Docker image** (from the project root):
    ```sh
    docker build -t nexa-visualize .
    ```
2. **Run the container**:
    ```sh
    docker run -p 3000:3000 nexa-visualize
    ```
3. Open your browser and go to: **http://localhost:3000**

---

### Option 2: Using Python

If you have Python installed, it comes with a simple built-in web server.

1.  Open a terminal or command prompt.
2.  Navigate to the `public` directory of this project:
    ```sh
    cd path/to/nexa-visualize/public
    ```
3.  Start the server.
    *   For **Python 3**: `python -m http.server`
    *   For **Python 2**: `python -m SimpleHTTPServer`

4.  Open your web browser and go to: **http://localhost:8000**

### Option 3: Using Node.js (`serve` package)

If you have Node.js and npm, you can use the `serve` package.

1.  Install `serve` globally (you only need to do this once):
    ```sh
    npm install -g serve
    ```
2.  In your terminal, navigate to the `public` directory:
    ```sh
    cd path/to/nexa-visualize/public
    ```
3.  Start the server:
    ```sh
    serve
    ```
4.  The terminal will give you a URL, usually `http://localhost:3000`. Open it in your browser.

### Option 4: Using VS Code Live Server Extension

If you use Visual Studio Code as your editor:

1.  Install the **Live Server** extension from the VS Code Marketplace.
2.  Open the `nexa-visualize` project folder in VS Code.
3.  In the file explorer, right-click on `public/index.html`.
4.  Select "Open with Live Server".
5.  Your browser will automatically open with the correct page.

## Technology Stack

-   **HTML5**
-   **CSS3**
-   **JavaScript (ES6+)**
-   **Three.js** for 3D rendering
