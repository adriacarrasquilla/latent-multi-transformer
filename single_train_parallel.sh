#!/bin/bash

# words="Bald,Attractive,Male,Wearing_Lipstick,Smiling,Eyeglasses,Big_Lips,High_Cheekbones,Young,Gray_Hair,No_Beard,Blond_Hair,Big_Nose,Bushy_Eyebrows,Pale_Skin,Chubby,Narrow_Eyes,Wavy_Hair,Bangs,Arched_Eyebrows"
words="Bald,Attractive,Male"
# words="Bald"

# Split the words into an array
IFS=',' read -ra word_array <<< "$words"

# Function to execute the Python script for each word
execute_python_script() {
    local word="$1"
    python single_train_parallel.py --attr "$word" --config single_parallel
}

start_time=$(date +%s)  # Start the timer

# Iterate over the word array and execute the Python script in parallel
for word in "${word_array[@]}"; do
    execute_python_script "$word" &
done

# Wait for all background processes to finish
wait

end_time=$(date +%s)  # Stop the timer

elapsed_time=$((end_time - start_time))  # Calculate the elapsed time

echo "Elapsed time: $elapsed_time seconds"
