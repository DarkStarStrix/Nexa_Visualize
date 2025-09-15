# Nexa Visualize

This is a 3D neural network visualization tool built with HTML, CSS, and Three.js.

## How to Run

To view this project, you need to run it from a local web server. You cannot simply open `public/index.html` as a file in your browser due to security policies (CORS).

Here are a few easy ways to start a local server:

### Option 1: Using Python

If you have Python installed, it comes with a simple built-in web server.

1.  Open a terminal or command prompt.
2.  Navigate to the `public` directory of this project:
    ```sh
    cd C:\Users\kunya\PycharmProjects\Nexa_Visualize\nexa-visualize\public
    ```
3.  Start the server.
    *   For **Python 3**: `python -m http.server`
    *   For **Python 2**: `python -m SimpleHTTPServer`

4.  Open your web browser and go to: **http://localhost:8000**

### Option 2: Using Node.js (`serve` package)

If you have Node.js and npm, you can use the `serve` package.

1.  Install `serve` globally (you only need to do this once):
    ```sh
    npm install -g serve
    ```
2.  In your terminal, navigate to the `public` directory:
    ```sh
    cd C:\Users\kunya\PycharmProjects\Nexa_Visualize\nexa-visualize\public
    ```
3.  Start the server:
    ```sh
    serve
    ```
4.  The terminal will give you a URL, usually `http://localhost:3000`. Open it in your browser.

### Option 3: Using VS Code Live Server Extension

If you use Visual Studio Code as your editor:

1.  Install the **Live Server** extension from the VS Code Marketplace.
2.  Open the `nexa-visualize` project folder in VS Code.
3.  In the file explorer, right-click on `public/index.html`.
4.  Select "Open with Live Server".
5.  Your browser will automatically open with the correct page.
