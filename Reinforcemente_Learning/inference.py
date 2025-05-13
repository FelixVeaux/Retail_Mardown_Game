# Phase 5: Final Model Packaging and Deployment Prep (Inference Module)
import torch
import numpy as np
import os

# Assuming other modules are accessible
import model_training # For model architecture, device
import data_preparation # For state encoding, price levels, constants

class InferenceModule:
    """Loads the trained RL model and provides an interface for inference."""
    def __init__(self, model_path=model_training.FINAL_MODEL_FILE,
                 device=model_training.DEVICE):
        self.device = device
        self.model = self._load_inference_model(model_path)
        self.price_levels = data_preparation.PRICE_LEVELS

    def _load_inference_model(self, model_path):
        """Loads the Q-network from the specified path."""
        model = model_training.QNetwork(model_training.STATE_DIM,
                                        model_training.ACTION_DIM, seed=0).to(self.device)
        try:
            # Load the state dict; ensure it's mapped to the correct device
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval() # Set to evaluation mode
            print(f"Inference model loaded successfully from {model_path}")
            return model
        except FileNotFoundError:
            print(f"Error: Inference model file not found at {model_path}")
            return None
        except Exception as e:
            print(f"Error loading inference model: {e}")
            return None

    def _get_valid_actions_mask(self, current_price):
        """Generates the valid action mask based on the current price.
           (Mirrors the logic from the environment but uses raw price).
        """
        num_actions = model_training.ACTION_DIM
        mask = np.zeros(num_actions, dtype=np.int8)

        current_price_rounded = round(current_price, 2)

        # Action 0 (keep price) is always initially considered valid
        mask[0] = 1

        # Check drop actions
        target_prices = {1: 54.0, 2: 48.0, 3: 36.0}
        for action_idx, target_price in target_prices.items():
            if current_price_rounded > target_price:
                mask[action_idx] = 1
            # If current price is already at or below the target, that specific drop action is invalid
            # Keeping the price (action 0) would be the valid way to stay at that price.

        return mask

    def predict_action(self, week, inventory, current_price, lift_tier):
        """Takes current state features, encodes them, and predicts the best action.

        Args:
            week (int): Current week number (1-15).
            inventory (float/int): Current remaining inventory.
            current_price (float): Current price level (e.g., 60.0, 54.0, ...).
            lift_tier (int): The lift tier category for the item (1, 2, or 3).

        Returns:
            int: The index of the recommended action (0-3), or -1 if prediction fails.
        """
        if self.model is None:
            print("Error: Inference model is not loaded.")
            return -1 # Indicate error

        # --- Input Validation (Basic) ---
        if not (1 <= week <= data_preparation.N_WEEKS):
             print(f"Warning: Week {week} is outside the expected range [1, {data_preparation.N_WEEKS}].")
             # Allow prediction but be aware it might be out of distribution
        if inventory < 0:
             print("Warning: Inventory is negative. Using 0 for prediction.")
             inventory = 0
        if round(current_price, 2) not in self.price_levels:
             print(f"Warning: Price {current_price} is not one of the standard levels {self.price_levels}. Encoding might be inaccurate.")
             # Find closest price or handle error? For now, allow data_preparation.encode_state to handle it.
        if lift_tier not in [1, 2, 3]:
             print(f"Warning: Lift tier {lift_tier} is not in the expected range [1, 2, 3].")
             # Allow prediction but be aware

        # --- State Encoding ---
        try:
            discretized_inventory = data_preparation.discretize_inventory(inventory)
            state_encoded = data_preparation.encode_state(week, discretized_inventory, current_price, lift_tier)
        except Exception as e:
            print(f"Error encoding state: {e}")
            return -1

        # --- Action Prediction --- #
        state_tensor = torch.from_numpy(state_encoded).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)

        # --- Action Masking --- #
        valid_actions_mask = self._get_valid_actions_mask(current_price)
        masked_action_values = action_values.cpu().data.numpy()[0].copy()
        masked_action_values[valid_actions_mask == 0] = -np.inf

        # --- Select Best Valid Action --- #
        predicted_action = np.argmax(masked_action_values).astype(int)

        return predicted_action

    def get_action_price(self, action_index):
        """Returns the target price corresponding to an action index."""
        if action_index == 0:
            # 'Keep price' - the price itself isn't determined by this action index alone
            # The caller should know the current price.
            # For clarity, we might return None or a special value.
             return None # Or raise ValueError("Action 0 (keep) doesn't define a target price.")
        elif action_index == 1:
            return 54.0
        elif action_index == 2:
            return 48.0
        elif action_index == 3:
            return 36.0
        else:
            return None # Invalid action index

# Example Usage (for testing)
if __name__ == '__main__':
    print("Running inference.py standalone example...")

    # Assume the final model exists
    model_file = model_training.FINAL_MODEL_FILE
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        print("Please ensure the model is trained and saved first (e.g., by running main.py or model_training.py).")
    else:
        # Initialize the inference module
        inference_engine = InferenceModule(model_path=model_file)

        if inference_engine.model:
            # Example state inputs
            week = 5
            inventory = 1550
            price = 60.0
            tier = 1

            print(f"\nExample Input State:")
            print(f"  Week: {week}")
            print(f"  Inventory: {inventory}")
            print(f"  Current Price: ${price:.2f}")
            print(f"  Lift Tier: {tier}")

            # Get recommended action
            recommended_action_idx = inference_engine.predict_action(week, inventory, price, tier)

            if recommended_action_idx != -1:
                print(f"\nRecommended Action Index: {recommended_action_idx}")

                # Get the corresponding price (if applicable)
                recommended_price = inference_engine.get_action_price(recommended_action_idx)
                if recommended_price is not None:
                    print(f"Implied Next Price: ${recommended_price:.2f}")
                else:
                    # Action 0 means keep current price
                    print(f"Action implies keeping the current price (${price:.2f})")
            else:
                print("\nFailed to get recommendation.")

            # --- Another Example ---
            week = 12
            inventory = 400
            price = 54.0 # Current price is 54
            tier = 2
            print(f"\nExample Input State:")
            print(f"  Week: {week}")
            print(f"  Inventory: {inventory}")
            print(f"  Current Price: ${price:.2f}")
            print(f"  Lift Tier: {tier}")
            recommended_action_idx = inference_engine.predict_action(week, inventory, price, tier)
            if recommended_action_idx != -1:
                 print(f"\nRecommended Action Index: {recommended_action_idx}")
                 recommended_price = inference_engine.get_action_price(recommended_action_idx)
                 if recommended_price is not None:
                     print(f"Implied Next Price: ${recommended_price:.2f}")
                 else:
                     print(f"Action implies keeping the current price (${price:.2f})")
            else:
                 print("\nFailed to get recommendation.")

    print("\nStandalone inference example finished.") 