def answer_step(self, current_probs, latent_z, prev_logits, step_emb):
        # 1. Fuse Inputs
        x = y_embed + latent_z + pos_embed + step_emb
        
        # 2. Process (The "Brain")
        for layer in self.layers:
            x = layer(x, context=x)
        x = self.norm(x)

        # 3. Generate Candidate (New Hypothesis)
        # We assume this is a fresh prediction of what the logits should be
        candidate_logits = self.output_head(x)
        
        # 4. Generate Gate (Confidence Switch)
        # Sigmoid forces range [0, 1]
        # 0 = Keep Memory (Prev), 1 = Overwrite (Candidate)
        gate_logit = self.gate_head(x) 
        
        # TIP: Initialize the gate bias to negative (e.g., -2.0).
        # This starts the gate near 0 (closed), forcing the model to 
        # initially rely on the previous step and only update when confident.
        gate = torch.sigmoid(gate_logit)

        # 5. GRU-style Update
        new_logits = (1 - gate) * prev_logits + gate * candidate_logits
        
        return new_logits