            # 1. Generate completions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            decoded_outputs = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            true_ans = example['answer'].split("####")[-1].strip()
            
            # 2. Multi-reward evaluation matching GDPO paper math
            reward_matrix = []
            for text in decoded_outputs:
                response_part = text[len(prompt_text):].strip()
                match = re.findall(r'\\boxed\{([^}]+)\}', response_part)
                
                r_format = 1.0 if match else 0.0
                r_correct = 1.0 if (match and match[-1].strip() == true_ans) else 0.0
                ans_str = match[-1].strip() if match else ""
                r_integer = 1.0 if ans_str.isdigit() else 0.0
                
                reward_matrix.append([r_format, r_correct, r_integer])

            # Convert to a tensor of shape [num_gens, 3]
            rewards_tensor = torch.tensor(reward_matrix, dtype=torch.float32, device="cuda")

            # GDPO Core Math: Decoupled Group-wise Normalization per column/reward
            mean = rewards_tensor.mean(dim=0, keepdim=True)
            std = rewards_tensor.std(dim=0, unbiased=False, keepdim=True) + 1e-8
            normalized_rewards = (rewards_tensor - mean) / std

            # Aggregate decoupled normalized rewards into final advantages per completion
            advantages = normalized_rewards.sum(dim=-1)

            # Pick best and worst based on true decoupled advantages
            best_idx = torch.argmax(advantages).item()
            worst_idx = torch.argmin(advantages).item()

            # 3. Optimization step with Quantized Reference Cache
            if best_idx != worst_idx and advantages[best_idx] > advantages[worst_idx]:
                best_response = decoded_outputs[best_idx][len(prompt_text):].strip()
                worst_response = decoded_outputs[worst_idx][len(prompt_text):].strip()
                
                self.optimizer.zero_grad()
                
                policy_chosen = self._compute_log_prob(prompt_text, best_response)
                policy_rejected = self._compute_log_prob(prompt_text, worst_response)
                
                with torch.no_grad():
                    ref_logps_dummy = torch.tensor([policy_chosen.item(), policy_rejected.item()], device="cuda")
                    q_chosen_cache = QuantizedReferenceCache.compress(ref_logps_dummy)
                    q_rejected_cache = QuantizedReferenceCache.compress(ref_logps_dummy)
                
                loss = qgdpo_loss(
                    policy_chosen=policy_chosen,
                    policy_rejected=policy_rejected,
                    ref_chosen_q=q_chosen_cache,
                    ref_rejected_q=q_rejected_cache,
                    beta=self.beta
                )
                
                loss.backward()
                self.optimizer.step()
                
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    
                print(f"Step {step}/{steps} | QGDPO Loss: {loss.item():.4f} (Advantage Gap: {(advantages[best_idx] - advantages[worst_idx]).item():.4f})")
            else:
                print(f"Step {step}/{steps} | Skipped (No decoupled advantage variance)")
