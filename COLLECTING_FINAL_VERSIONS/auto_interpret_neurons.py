import pandas as pd
import numpy as np
import torch
import argparse
import sys
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import anthropic
from scipy.stats import pearsonr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NeuronInterpreter:
    """
    Automated neuron interpretation using Claude API.
    """
    
    def __init__(self, tsv_path: str, binary_annotations_path: str, activations_path: str, 
                 claude_api_key: str, output_dir: str = "neuron_interpretations", 
                 normalize_activations: bool = True, num_neurons: int = 10, x: int = 20, y: int = 20, z: int = 20,
                 neuron_indices: Optional[List[int]] = None, excluded_attributes: Optional[List[str]] = None,
                 command_used: Optional[str] = None, random_seed: int = 42, ratio: float = 1.0):
        """
        Initialize the neuron interpreter.
        
        Args:
            tsv_path: Path to the original UniProt TSV file
            binary_annotations_path: Path to the binary annotations CSV file
            activations_path: Path to the .pt file containing neuron activations
            claude_api_key: Claude API key
            output_dir: Directory to save results (default: "neuron_interpretations")
            normalize_activations: Whether to normalize activations 0-10 per neuron (default: True)
            num_neurons: Number of neurons to interpret (used if neuron_indices not provided)
            x: Number of top activating proteins to select
            y: Number of random non-activating proteins to select
            z: Number of random activating proteins to select (not in top x)
            neuron_indices: Optional explicit list of neuron indices to interpret
            excluded_attributes: Optional list of attribute names to exclude from interpretations
            command_used: Optional command string that was used to run the script
            random_seed: Random seed for reproducible protein selection and other random choices (default: 42)
            ratio: Minimum ratio of available proteins to required proteins for interpretation (default: 1.0)
        """
        self.tsv_path = tsv_path
        self.binary_annotations_path = binary_annotations_path
        self.activations_path = activations_path
        self.claude_api_key = claude_api_key
        self.normalize_activations = normalize_activations
        self.num_neurons = num_neurons
        self.neuron_indices = neuron_indices
        self.x = x
        self.y = y
        self.z = z
        self.random_seed = random_seed
        self.ratio = ratio
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Set excluded attributes
        if excluded_attributes is None:
            excluded_attributes = []
        self.excluded_attributes = set(excluded_attributes)
        
        # Store the command used
        self.command_used = command_used
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(output_dir)
        self.output_dir = base_output_dir / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the command used if provided
        if self.command_used:
            command_file = self.output_dir / "command_used.txt"
            with open(command_file, 'w') as f:
                f.write(f"Command executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n")
                f.write(f"{self.command_used}\n")
        
        # Initialize data containers
        self.tsv_data = None
        self.binary_annotations = None
        self.activations = None
        
        print(f"Initialized NeuronInterpreter:")
        print(f"  - TSV file: {tsv_path}")
        print(f"  - Binary annotations: {binary_annotations_path}")
        print(f"  - Activations file: {activations_path}")
        print(f"  - Normalize activations: {normalize_activations}")
        if self.neuron_indices is not None and len(self.neuron_indices) > 0:
            print(f"  - Specified neurons to interpret: {self.neuron_indices}")
        else:
            print(f"  - Number of neurons to interpret: {num_neurons}")
        print(f"  - Sample sizes: x={x}, y={y}, z={z}")
        print(f"  - Random seed: {random_seed}")
        print(f"  - Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load all required data files."""
        print("\nLoading data files...")
        
        # Load TSV data
        print("Loading TSV file...")
        self.tsv_data = pd.read_csv(self.tsv_path, sep='\t', low_memory=False)
        print(f"  Loaded {len(self.tsv_data)} proteins with {len(self.tsv_data.columns)} columns")
        
        # Load binary annotations
        print("Loading binary annotations...")
        self.binary_annotations = pd.read_csv(self.binary_annotations_path)
        print(f"  Loaded {len(self.binary_annotations)} proteins with {len(self.binary_annotations.columns)} columns")
        
        # Load activations
        print("Loading activations...")
        loaded_data = torch.load(self.activations_path, map_location='cpu')
        
        # Handle different formats of saved activations
        if isinstance(loaded_data, dict):
            # New format from extract_protein_embeddings.py
            if 'embeddings' in loaded_data:
                self.activations = loaded_data['embeddings']
                print(f"  Loaded embeddings from dictionary format")
                if 'processing_info' in loaded_data:
                    info = loaded_data['processing_info']
                    print(f"  Processing info: {info.get('n_proteins_processed', 'unknown')} proteins, "
                          f"hidden_dim={info.get('hidden_dim', 'unknown')}")
            else:
                # Dictionary format but no 'embeddings' key - try to find the tensor
                tensor_keys = [k for k, v in loaded_data.items() if isinstance(v, torch.Tensor)]
                if tensor_keys:
                    print(f"  Found tensor keys: {tensor_keys}")
                    # Use the first tensor found
                    self.activations = loaded_data[tensor_keys[0]]
                    print(f"  Using tensor from key: {tensor_keys[0]}")
                else:
                    raise ValueError(f"Could not find tensor data in loaded dictionary. Keys: {list(loaded_data.keys())}")
        elif isinstance(loaded_data, torch.Tensor):
            # Direct tensor format
            self.activations = loaded_data
            print(f"  Loaded direct tensor format")
        else:
            raise ValueError(f"Unexpected data format: {type(loaded_data)}")
        
        # Convert to numpy if it's a tensor
        if isinstance(self.activations, torch.Tensor):
            self.activations = self.activations.numpy()
        
        print(f"  Loaded activations with shape: {self.activations.shape}")
        
        # Validate data consistency
        self._validate_data()
        
        # Normalize activations if requested
        if self.normalize_activations:
            self._normalize_activations()
        
    def _validate_data(self):
        """Validate that all data files are consistent."""
        print("\nValidating data consistency...")
        
        # Check that number of proteins matches across files
        n_tsv = len(self.tsv_data)
        n_binary = len(self.binary_annotations)
        n_activations = self.activations.shape[0]
        
        if not (n_tsv == n_binary == n_activations):
            raise ValueError(f"Mismatch in number of proteins: TSV={n_tsv}, Binary={n_binary}, Activations={n_activations}")
        
        print(f"  ✅ All files have {n_tsv} proteins")
        print(f"  ✅ Activations shape: {self.activations.shape} (proteins x neurons)")
        
    def _normalize_activations(self):
        """
        Normalize activations using min-max normalization per neuron.
        For each neuron, scales activations to range [0, 10].
        If a neuron never activates (min == max), both min and max remain 0.
        """
        print("\nNormalizing activations...")
        
        n_proteins, n_neurons = self.activations.shape
        normalized_activations = np.zeros_like(self.activations)
        
        neurons_normalized = 0
        neurons_constant = 0
        
        for neuron_idx in range(n_neurons):
            neuron_activations = self.activations[:, neuron_idx]
            
            min_val = np.min(neuron_activations)
            max_val = np.max(neuron_activations)
            
            if min_val == max_val:
                # Neuron has constant activation (including all zeros)
                # Keep as is (will be all the same value, likely 0)
                normalized_activations[:, neuron_idx] = neuron_activations
                neurons_constant += 1
            else:
                # Apply min-max normalization to scale to [0, 10]
                normalized_activations[:, neuron_idx] = 10 * (neuron_activations - min_val) / (max_val - min_val)
                neurons_normalized += 1
        
        self.activations = normalized_activations
        print(f"  ✅ Normalized {neurons_normalized} neurons to [0, 10] range")
        if neurons_constant > 0:
            print(f"  ⚠️  {neurons_constant} neurons had constant activations (kept unchanged)")
        
    def select_proteins_for_neuron(self, neuron_idx: int) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """
        Select proteins for a given neuron: top x activating, y random non-activating, z random activating.
        
        Args:
            neuron_idx: Index of the neuron
            
        Returns:
            Tuple of (top_activating_indices, random_non_activating_indices, random_activating_indices)
            or None if there aren't enough proteins available
        """
        neuron_activations = self.activations[:, neuron_idx]
        
        # Define activation threshold as 0
        activation_threshold = 0.0
        
        # Get indices sorted by activation (highest first)
        sorted_indices = np.argsort(neuron_activations)[::-1]
        
        # Top x activating proteins
        top_activating = sorted_indices[:self.x].tolist()
        
        # Non-activating proteins (below threshold)
        non_activating_indices = np.where(neuron_activations <= activation_threshold)[0]
        non_activating_indices = [idx for idx in non_activating_indices if idx not in top_activating]
        
        # Activating proteins (above threshold, excluding top x)
        activating_indices = np.where(neuron_activations > activation_threshold)[0]
        activating_indices = [idx for idx in activating_indices if idx not in top_activating]
        
        # Check if we have enough proteins (with ratio multiplier for stricter requirements)
        total_activating_needed = self.x + self.z
        total_activating_required = int(self.ratio * total_activating_needed)
        total_activating_available = len(top_activating) + len(activating_indices)
        
        if total_activating_available < total_activating_required:
            print(f"Skipping neuron {neuron_idx}: Only {total_activating_available} activating proteins available, need {total_activating_required} (ratio={self.ratio} * {total_activating_needed})")
            return None
        
        non_activating_required = int(self.ratio * self.y)
        if len(non_activating_indices) < non_activating_required:
            print(f"Skipping neuron {neuron_idx}: Only {len(non_activating_indices)} non-activating proteins available, need {non_activating_required} (ratio={self.ratio} * {self.y})")
            return None
        
        # Random y non-activating proteins
        random_non_activating = random.sample(non_activating_indices, self.y)
        
        # Random z activating proteins
        random_activating = random.sample(activating_indices, self.z)
        
        return top_activating, random_non_activating, random_activating
    
    def get_protein_annotations(self, protein_idx: int) -> Dict[str, str]:
        """
        Get all annotations for a protein based on binary annotations.
        
        Args:
            protein_idx: Index of the protein
            
        Returns:
            Dictionary of annotation_name: annotation_value
        """
        annotations = {}
        
        # Get binary annotation row
        binary_row = self.binary_annotations.iloc[protein_idx]
        tsv_row = self.tsv_data.iloc[protein_idx]
        
        # Add identifier information
        identifier_columns = ['Entry', 'Entry Name', 'Protein names', 'Gene Names', 'Organism']
        for col in identifier_columns:
            if col in tsv_row.index and pd.notna(tsv_row[col]) and col not in self.excluded_attributes:
                annotations[col] = str(tsv_row[col])
        
        # Get annotation columns that have value 1 in binary annotations
        annotation_cols = [col for col in binary_row.index if col.startswith('has_') and binary_row[col] == 1]
        
        for binary_col in annotation_cols:
            # Map back to original column name
            original_col = self._map_binary_to_original_column(binary_col)
            if (original_col and original_col in tsv_row.index and 
                pd.notna(tsv_row[original_col]) and original_col not in self.excluded_attributes):
                annotations[original_col] = str(tsv_row[original_col])
        
        return annotations
    
    def _map_binary_to_original_column(self, binary_col_name: str) -> str:
        """Map binary column name back to original column name."""
        # Remove 'has_' prefix and reverse transformations
        original = binary_col_name.replace('has_', '', 1)
        original = original.replace('_', ' ')
        
        # Common mappings
        mapping = {
            'gene ontology ids': 'Gene Ontology IDs',
            'gene ontology biological process': 'Gene Ontology (biological process)',
            'gene ontology cellular component': 'Gene Ontology (cellular component)',
            'gene ontology molecular function': 'Gene Ontology (molecular function)',
            'gene ontology go': 'Gene Ontology (GO)',
            'protein families': 'Protein families',
            'ec number': 'EC number',
            'function cc': 'Function [CC]',
            'sequence similarities': 'Sequence similarities',
            'zinc finger': 'Zinc finger',
            'compositional bias': 'Compositional bias',
            'domain cc': 'Domain [CC]',
            'domain ft': 'Domain [FT]',
            'subcellular location cc': 'Subcellular location [CC]',
            'topological domain': 'Topological domain',
            'involvement in disease': 'Involvement in disease',
            'pharmaceutical use': 'Pharmaceutical use',
            'pubmed id': 'PubMed ID',
            'doi id': 'DOI ID',
            'beta strand': 'Beta strand',
            'active site': 'Active site',
            'binding site': 'Binding site',
            'catalytic activity': 'Catalytic activity',
            'dna binding': 'DNA binding',
            'ph dependence': 'pH dependence',
            'activity regulation': 'Activity regulation',
            'redox potential': 'Redox potential',
            'rhea id': 'Rhea ID',
            'temperature dependence': 'Temperature dependence'
        }
        
        return mapping.get(original.lower(), original)
    
    def create_interpretation_prompt(self, protein_indices: List[int], neuron_idx: int) -> str:
        """
        Create prompt for Claude to interpret neuron function.
        
        Args:
            protein_indices: List of protein indices
            neuron_idx: Neuron index
            
        Returns:
            Prompt string
        """
        prompt = f"I'm studying a neuron (neuron #{neuron_idx}) from a neural network trained on protein data. "
        prompt += f"Below are {len(protein_indices)} proteins with their annotations and activation values for this neuron.\n\n"
        prompt += "The proteins are a mix of:\n"
        prompt += "- Activating proteins (proteins that activate this neuron)\n"
        prompt += "- Non-activating proteins (proteins that don't activate this neuron)\n\n"
        
        prompt += "For each protein, I'm providing:\n"
        prompt += "- All available biological annotations\n"
        prompt += "- Activation value for this neuron\n\n"
        
        prompt += "PROTEINS:\n"
        prompt += "=" * 50 + "\n\n"
        
        for i, protein_idx in enumerate(protein_indices):
            activation_value = self.activations[protein_idx, neuron_idx]
            annotations = self.get_protein_annotations(protein_idx)
            
            prompt += f"PROTEIN {i+1} (Activation: {activation_value:.4f}):\n"
            
            for ann_name, ann_value in annotations.items():
                prompt += f"  {ann_name}: {ann_value}\n"
            
            prompt += "\n" + "-" * 30 + "\n\n"
        
        prompt += "There is something that is causing this neuron to fire. Based on what you observe in the annotations and activation values, please provide an interpretation of what this neuron appears to be responding to. Your response should be no longer than a single sentence."
        
        return prompt
    
    def create_prediction_prompt(self, protein_indices: List[int], interpretation: str) -> str:
        """
        Create prompt for Claude to predict activation values based on interpretation.
        
        Args:
            protein_indices: List of protein indices
            interpretation: Previously generated interpretation
            
        Returns: 
            Prompt string
        """
        prompt = f"This is an interpretation of a neuron: '{interpretation}'\n\n"
        prompt += f"Below are {len(protein_indices)} proteins with their annotations. "
        prompt += "Please predict how strongly this neuron would activate for each protein using the interpretation of the neuron written above.\n\n"
        prompt += "Provide activation values as numbers between 0 and 10, where:\n"
        prompt += "- 0 = no activation (protein doesn't correspond to the interpretation)\n"
        prompt += "- 10 = maximum activation (protein strongly corresponds to the interpretation)\n\n"
        
        prompt += "PROTEINS:\n"
        prompt += "=" * 50 + "\n\n"
        
        for i, protein_idx in enumerate(protein_indices):
            annotations = self.get_protein_annotations(protein_idx)
            
            prompt += f"PROTEIN {i+1}:\n"
            
            for ann_name, ann_value in annotations.items():
                prompt += f"  {ann_name}: {ann_value}\n"
            
            prompt += "\n" + "-" * 30 + "\n\n"
        
        prompt += f"Your response should be exactly {len(protein_indices)} numbers (one per protein, in order), "
        prompt += "separated by spaces or newlines. Do not include any other text, explanations, or formatting."
        
        return prompt
    
    def call_claude_api(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call Claude API with the given prompt.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            
        Returns:
            Claude's response
        """
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                # tools = [{
                #     "type": "function",
                #     "function": {
                #         "type": "web_search_20250305",
                #         "name": "web_search",
                #         "description": "A tool to search the web. Can be used to look up a DOI ID.",
                #     }
                # }]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return None
    
    def parse_prediction_response(self, response: str, expected_count: int) -> List[float]:
        """
        Parse Claude's prediction response to extract numeric values.
        
        Args:
            response: Claude's response
            expected_count: Expected number of values
            
        Returns:
            List of predicted activation values
        """
        if not response:
            return [0.0] * expected_count
        
        # Extract numbers from response
        import re
        numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', response)
        
        try:
            predictions = [float(num) for num in numbers[:expected_count]]
            
            # Ensure we have the right number of predictions
            while len(predictions) < expected_count:
                predictions.append(0.0)
            
            return predictions[:expected_count]
        except ValueError:
            print(f"Warning: Could not parse predictions from response: {response[:100]}...")
            return [0.0] * expected_count
    
    def interpret_neuron(self, neuron_idx: int) -> Optional[Dict]:
        """
        Interpret a single neuron.
        
        Args:
            neuron_idx: Index of the neuron to interpret
            
        Returns:
            Dictionary with interpretation results, or None if neuron was skipped
        """
        print(f"\nInterpreting neuron {neuron_idx}...")
        
        # Select proteins
        protein_selection = self.select_proteins_for_neuron(neuron_idx)
        if protein_selection is None:
            return None
        
        top_activating, random_non_activating, random_activating = protein_selection
        
        # Combine activating proteins (top + random activating)
        all_activating = top_activating + random_activating
        
        # Split activating proteins evenly between train and test
        mid_activating = len(all_activating) // 2
        train_activating = all_activating[:mid_activating]
        test_activating = all_activating[mid_activating:]
        
        # Split non-activating proteins evenly between train and test
        mid_non_activating = len(random_non_activating) // 2
        train_non_activating = random_non_activating[:mid_non_activating]
        test_non_activating = random_non_activating[mid_non_activating:]
        
        # Combine and shuffle within each set (so prompts don't show all active then all inactive)
        set1_proteins = train_activating + train_non_activating
        random.shuffle(set1_proteins)
        
        set2_proteins = test_activating + test_non_activating
        random.shuffle(set2_proteins)
        
        print(f"  Selected {len(set1_proteins)} proteins for interpretation ({len(train_activating)} active, {len(train_non_activating)} inactive)")
        print(f"  Selected {len(set2_proteins)} proteins for testing ({len(test_activating)} active, {len(test_non_activating)} inactive)")
        
        # Step 1: Get interpretation from first set
        interpretation_prompt = self.create_interpretation_prompt(set1_proteins, neuron_idx)
        interpretation_response = self.call_claude_api(interpretation_prompt)
        
        if not interpretation_response:
            print(f"  Failed to get interpretation for neuron {neuron_idx}")
            return None
        
        interpretation = interpretation_response.strip()
        print(f"  Interpretation: {interpretation}")
        
        # Step 2: Get predictions for second set
        prediction_prompt = self.create_prediction_prompt(set2_proteins, interpretation)
        prediction_response = self.call_claude_api(prediction_prompt, max_tokens=200)
        
        if not prediction_response:
            print(f"  Failed to get predictions for neuron {neuron_idx}")
            return None
        
        # Parse predictions
        predicted_activations = self.parse_prediction_response(prediction_response, len(set2_proteins))
        actual_activations = [self.activations[idx, neuron_idx] for idx in set2_proteins]
        
        # Calculate correlation
        if len(predicted_activations) > 1 and len(actual_activations) > 1:
            correlation, p_value = pearsonr(predicted_activations, actual_activations)
        else:
            correlation, p_value = 0.0, 1.0
        
        print(f"  Correlation: {correlation:.4f} (p={p_value:.4f})")
        
        # Save prompts to file
        prompt_file = self.output_dir / f"neuron_{neuron_idx}_prompts.txt"
        with open(prompt_file, 'w') as f:
            f.write("INTERPRETATION PROMPT:\n")
            f.write("=" * 50 + "\n")
            f.write(interpretation_prompt)
            f.write("\n\nINTERPRETATION RESPONSE:\n")
            f.write("=" * 50 + "\n")
            f.write(interpretation_response)
            f.write("\n\n" + "=" * 80 + "\n\n")
            f.write("PREDICTION PROMPT:\n")
            f.write("=" * 50 + "\n")
            f.write(prediction_prompt)
            f.write("\n\nPREDICTION RESPONSE:\n")
            f.write("=" * 50 + "\n")
            f.write(prediction_response)
        
        return {
            'neuron_idx': neuron_idx,
            'interpretation': interpretation,
            'correlation': correlation,
            'p_value': p_value,
            'predicted_activations': predicted_activations,
            'actual_activations': actual_activations,
            'test_proteins': set2_proteins
        }
    
    def run_interpretation(self):
        """Run interpretation for all selected neurons."""
        if self.neuron_indices is not None and len(self.neuron_indices) > 0:
            print(f"\nStarting interpretation of {len(self.neuron_indices)} specified neurons...")
        else:
            print(f"\nStarting interpretation of {self.num_neurons} neurons...")
        
        # Load data
        self.load_data()
        
        # Select neurons to interpret
        total_neurons = self.activations.shape[1]
        if self.neuron_indices is not None and len(self.neuron_indices) > 0:
            # Validate and de-duplicate provided indices
            unique_indices = []
            seen = set()
            for idx in self.neuron_indices:
                if idx in seen:
                    continue
                if 0 <= idx < total_neurons:
                    unique_indices.append(idx)
                    seen.add(idx)
                else:
                    print(f"Warning: Provided neuron index {idx} is out of range [0, {total_neurons-1}] and will be skipped")
            if len(unique_indices) == 0:
                raise ValueError("No valid neuron indices provided after validation")
            selected_neurons = unique_indices
            
            # Results storage
            results = []
            skipped_neurons = []
            
            # Process each specified neuron
            for neuron_idx in selected_neurons:
                result = self.interpret_neuron(neuron_idx)
                if result:
                    results.append(result)
                else:
                    skipped_neurons.append(neuron_idx)
        else:
            # Keep selecting neurons until we have enough successful interpretations
            if self.num_neurons > total_neurons:
                print(f"Warning: Requested {self.num_neurons} neurons but only {total_neurons} available")
                self.num_neurons = total_neurons
            
            # Results storage
            results = []
            skipped_neurons = []
            tried_neurons = set()
            
            print(f"Will keep selecting neurons until {self.num_neurons} are successfully interpreted...")
            
            while len(results) < self.num_neurons and len(tried_neurons) < total_neurons:
                # Select a random neuron that we haven't tried yet
                available_neurons = [i for i in range(total_neurons) if i not in tried_neurons]
                if not available_neurons:
                    break
                
                neuron_idx = random.choice(available_neurons)
                tried_neurons.add(neuron_idx)
                
                print(f"Trying neuron {neuron_idx} (attempt {len(tried_neurons)}, successful: {len(results)}/{self.num_neurons})...")
                
                result = self.interpret_neuron(neuron_idx)
                if result:
                    results.append(result)
                    print(f"✅ Successfully interpreted neuron {neuron_idx} ({len(results)}/{self.num_neurons} completed)")
                    
                    # Stop as soon as we have enough successful interpretations
                    if len(results) >= self.num_neurons:
                        break
                else:
                    skipped_neurons.append(neuron_idx)
            
            # Check if we couldn't find enough neurons
            if len(results) < self.num_neurons:
                print(f"⚠️  Could only interpret {len(results)} neurons out of {self.num_neurons} requested")
                print(f"   Tried {len(tried_neurons)} neurons total, {len(skipped_neurons)} were skipped due to insufficient proteins")
            
            selected_neurons = [result['neuron_idx'] for result in results] + skipped_neurons
        
        print(f"Final selection: {[result['neuron_idx'] for result in results]} (successful)")
        
        # Save results
        self._save_results(results, skipped_neurons)
        
        print(f"\n✅ Completed interpretation of {len(results)} neurons")
        if skipped_neurons:
            print(f"⚠️  Skipped {len(skipped_neurons)} neurons due to insufficient proteins: {skipped_neurons}")
        print(f"📁 Results saved to: {self.output_dir}")
    
    def _save_results(self, results: List[Dict], skipped_neurons: List[int] = None):
        """Save results to CSV and text files."""
        if not results and not skipped_neurons:
            print("No results to save")
            return
        
        if skipped_neurons is None:
            skipped_neurons = []
        
        # Calculate statistics
        if results:
            correlations = [result['correlation'] for result in results]
            # Use nanmean and nanmedian to ignore NaN values
            mean_correlation = np.nanmean(correlations)
            median_correlation = np.nanmedian(correlations)
        else:
            mean_correlation = np.nan
            median_correlation = np.nan
        
        # Create CSV with summary results
        csv_data = []
        for result in results:
            csv_data.append({
                'neuron_idx': result['neuron_idx'],
                'interpretation': result['interpretation'],
                'correlation': result['correlation'],
                'p_value': result['p_value']
            })
        
        # Add statistics rows to CSV
        csv_data.append({
            'neuron_idx': 'MEAN',
            'interpretation': '',
            'correlation': mean_correlation,
            'p_value': ''
        })
        csv_data.append({
            'neuron_idx': 'MEDIAN',
            'interpretation': '',
            'correlation': median_correlation,
            'p_value': ''
        })
        
        # Add skipped neurons information to CSV
        if skipped_neurons:
            csv_data.append({
                'neuron_idx': 'SKIPPED',
                'interpretation': f"Skipped neurons: {skipped_neurons}",
                'correlation': '',
                'p_value': ''
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "interpretation_results.csv"
        csv_df.to_csv(csv_file, index=False)
        print(f"📊 CSV results saved to: {csv_file}")
        
        # Create detailed text file
        text_file = self.output_dir / "interpretation_summary.txt"
        with open(text_file, 'w') as f:
            # Add command at the top if available
            if self.command_used:
                f.write(f"Command executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n")
                f.write(f"{self.command_used}\n")
                f.write("=" * 80 + "\n\n")
            
            f.write("NEURON INTERPRETATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            if results:
                for result in results:
                    f.write(f"NEURON {result['neuron_idx']}:\n")
                    f.write(f"Interpretation: {result['interpretation']}\n")
                    f.write(f"Correlation: {result['correlation']:.4f} (p={result['p_value']:.4f})\n")
                    f.write("-" * 30 + "\n\n")
            else:
                f.write("No neurons were successfully interpreted.\n\n")
            
            # Add statistics at the bottom
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 30 + "\n")
            if not np.isnan(mean_correlation):
                f.write(f"Mean interpretation score (correlation): {mean_correlation:.4f}\n")
                f.write(f"Median interpretation score (correlation): {median_correlation:.4f}\n")
            else:
                f.write("Mean interpretation score (correlation): N/A (no successful interpretations)\n")
                f.write("Median interpretation score (correlation): N/A (no successful interpretations)\n")
            f.write(f"Total neurons interpreted: {len(results)}\n")
            if skipped_neurons:
                f.write(f"Total neurons skipped: {len(skipped_neurons)}\n")
                f.write(f"Skipped neurons: {skipped_neurons}\n")
        
        print(f"📝 Summary saved to: {text_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated neuron interpretation using Claude API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_interpret_neurons.py -t data.tsv -b binary_annotations.csv -a activations.pt -k YOUR_API_KEY
  python auto_interpret_neurons.py -t data.tsv -b binary_annotations.csv -a activations.pt -k YOUR_API_KEY -o my_results
  python auto_interpret_neurons.py -t data.tsv -b binary_annotations.csv -a activations.pt -k YOUR_API_KEY -n 5 -x 15 -y 15 -z 15
  python auto_interpret_neurons.py -t data.tsv -b binary_annotations.csv -a activations.pt -k YOUR_API_KEY --neurons 0,5,42,77 -o custom_output_dir
  python auto_interpret_neurons.py -t data.tsv -b binary_annotations.csv -a activations.pt -k YOUR_API_KEY --no_normalize_activations
  python auto_interpret_neurons.py -t data.tsv -b binary_annotations.csv -a activations.pt -k YOUR_API_KEY --random_seed 123
        """
    )
    
    parser.add_argument("-t", "--tsv_file", required=True, help="Path to the UniProt TSV file")
    parser.add_argument("-b", "--binary_annotations", required=True, help="Path to the binary annotations CSV file")
    parser.add_argument("-a", "--activations", required=True, help="Path to the .pt file with neuron activations")
    parser.add_argument("-k", "--claude_api_key", required=True, help="Claude API key")
    parser.add_argument("-n", "--num_neurons", type=int, default=10, help="Number of neurons to interpret (default: 10)")
    parser.add_argument("-x", type=int, default=20, help="Number of top activating proteins (default: 20)")
    parser.add_argument("-y", type=int, default=20, help="Number of random non-activating proteins (default: 20)")
    parser.add_argument("-z", type=int, default=20, help="Number of random activating proteins (default: 20)")
    parser.add_argument("-o", "--output_dir", type=str, default="neuron_interpretations", help="Directory to save results (default: neuron_interpretations)")
    parser.add_argument("--normalize_activations", action="store_true", default=True, help="Normalize activations to [0,10] range per neuron (default: True)")
    parser.add_argument("--no_normalize_activations", dest="normalize_activations", action="store_false", help="Disable activation normalization")
    parser.add_argument("--neurons", type=str, default=None, help="Comma- or space-separated list of neuron indices to interpret (overrides -n)")
    parser.add_argument("--exclude_attributes", type=str, default=None, help="Comma-separated list of attribute names to exclude from interpretations (e.g., 'PubMed ID,DOI ID,Gene Ontology IDs')")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible protein selection and other random choices (default: 42)")
    parser.add_argument("--ratio", type=float, default=1.0, help="Minimum ratio of available proteins to required proteins for interpretation (default: 1.0). Higher values require more proteins to be available before attempting interpretation.")
    
    args = parser.parse_args()
    
    # Parse neuron indices if provided
    neuron_indices = None
    if args.neurons is not None:
        raw = args.neurons.strip()
        if raw:
            # Support comma and/or space separated
            separators = [',', ' ']
            for sep in separators:
                raw = raw.replace(sep, ' ')
            try:
                neuron_indices = [int(tok) for tok in raw.split() if tok]
            except ValueError:
                print(f"Error: --neurons must be a list of integers, got: '{args.neurons}'")
                sys.exit(1)
        if neuron_indices is not None and len(neuron_indices) == 0:
            neuron_indices = None
    
    # Parse excluded attributes if provided
    excluded_attributes = None
    if args.exclude_attributes is not None:
        raw = args.exclude_attributes.strip()
        if raw:
            # Split by comma and clean up whitespace
            excluded_attributes = [attr.strip() for attr in raw.split(',') if attr.strip()]
            print(f"Excluding attributes: {excluded_attributes}")
    
    # Validate input files
    for file_path, name in [(args.tsv_file, "TSV file"), (args.binary_annotations, "Binary annotations"), (args.activations, "Activations file")]:
        if not os.path.exists(file_path):
            print(f"Error: {name} '{file_path}' does not exist.")
            sys.exit(1)
    
    print("🧠 Neuron Interpretation Pipeline")
    print("=" * 40)
    
    try:
        # Reconstruct the command used
        command_used = " ".join(sys.argv)
        
        interpreter = NeuronInterpreter(
            tsv_path=args.tsv_file,
            binary_annotations_path=args.binary_annotations,
            activations_path=args.activations,
            claude_api_key=args.claude_api_key,
            output_dir=args.output_dir,
            normalize_activations=args.normalize_activations,
            num_neurons=args.num_neurons,
            x=args.x,
            y=args.y,
            z=args.z,
            neuron_indices=neuron_indices,
            excluded_attributes=excluded_attributes,
            command_used=command_used,
            random_seed=args.random_seed,
            ratio=args.ratio
        )
        
        interpreter.run_interpretation()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 