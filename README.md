To run the application, follow these steps:

1.  **Navigate to the `scripts` directory:** This directory contains the scripts for scraping, chunking, embedding, and creating the vector database.

    ```bash
    cd scripts
    ```

2.  **Run the scripts in the following order:**

    a.  **Scrape the data:**

        ```bash
        python scraper.py
        ```

    b.  **Chunk the scraped data:**

        ```bash
        python chunker.py
        ```

    c.  **Embed the chunks:**

        ```bash
        python embeder.py
        ```

    d.  **Create the vector database:**

        ```bash
        python vectordb.py
        ```

3.  **Navigate to the `src` directory:** This directory contains the main application file.

    ```bash
    cd ../src
    ```

4.  **Run the main application:**

    ```bash
    uv run main.py
    ```