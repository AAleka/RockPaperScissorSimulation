package main

import "core:fmt"
import "core:math"
import "core:time"
import "core:math/rand"

Entity :: struct {
	type : string,
	symbol : string,
	y : int,
	x : int,
}

print_field :: proc() {
	for i in 0 ..= HEIGHT-1 {
		for j in 0 ..= WIDTH-1 {
			if FIELD[i][j] == "P" {
				fmt.print("\033[31m")
			}
			else if FIELD[i][j] == "R" {
				fmt.print("\033[32m")
			}
			else if FIELD[i][j] == "S" {
				fmt.print("\033[34m")
			}
			else {
				fmt.print("\033[37m")
			}
			fmt.print(FIELD[i][j])
		}
		fmt.println()
	}
}	

reset_field :: proc() -> (Ns : [3]int) {
	FIELD = "·"
	
	for i in 0 ..= WIDTH-1 { 
		FIELD[0][i] = "‾"
		FIELD[HEIGHT-1][i] = "_"
	}

	for i in 0 ..= HEIGHT-1 { 
		FIELD[i][0] = "|"
		FIELD[i][WIDTH-1] = "|"
	}

	for e in Entities {
		FIELD[e.y][e.x] = e.symbol
		
		if e.symbol == "S" {
			Ns[0] += 1
		}
		else if e.symbol == "P" {
			Ns[1] += 1
		}
		else if e.symbol == "R" {
			Ns[2] += 1
		}
	}

	return
}

enemy_type :: proc(type : string) -> (string, string) {
	switch type {
		case "Scissor":
			return "P", "R"
		case "Paper":
			return "R", "S"
		case "Rock":
			return "S", "P"
	}
	
	return "Error", "Error"
}

closest_enemy :: proc(e : Entity) -> (min_y, min_x : int) {
	enemy_type, hunter_type := enemy_type(e.type)
	min_d : f32 = 100_000.0

	for i in 1 ..= HEIGHT-2 {
		for j in 1 ..= WIDTH-2 {
			if FIELD[i][j] == enemy_type {
				d := math.sqrt_f32(cast(f32)((e.y-i)*(e.y-i) + (e.x-j)*(e.x-j)))
				if min_d > d {
					min_d = d
					min_y, min_x = i, j
				}
			}
		}
	}

	return
}

closest_hunter :: proc(e : Entity) -> (min_y, min_x : int) {
	enemy_type, hunter_type := enemy_type(e.type)
	min_d : f32 = 100_000.0

	for i in 1 ..= HEIGHT-2 {
		for j in 1 ..= WIDTH-2 {
			if FIELD[i][j] == hunter_type {
				d := math.sqrt_f32(cast(f32)((e.y-i)*(e.y-i) + (e.x-j)*(e.x-j)))
				if min_d > d {
					min_d = d
					min_y, min_x = i, j
				}
			}
		}
	}

	return
}

relu :: proc(X : [N_neurons]f64) -> (A : [N_neurons]f64) {
	for i in 0 ..< len(A) {
		A[i] = math.max(0, X[i])
	}

	return
}

softmax :: proc(X : [N_outputs]f64) -> (softmax : [N_outputs]f64) {
	sum : f64 = 0.0

	for x in X {
		sum += math.exp_f64(x)
	}

	for i in 0 ..< N_outputs {
		softmax[i] = math.exp_f64(X[i]) / sum
	}

	return 
}

CrossEntropyLoss :: proc(x : f64) -> f64 {
	return -math.ln_f64(x)
}

NN_train :: proc(distance_target, distance_hunter, angle_target, angle_hunter : f32, best_action_idx : int, avg_loss : f64) -> f64 {
	alpha :: 0.0001
	epsylon :: 0.001
	max_d := math.sqrt_f64(cast(f64)(math.pow_f64((HEIGHT-2), 2) + math.pow_f64((WIDTH-2), 2)))

	X := [4]f64 {
		cast(f64)distance_target / max_d, 
		cast(f64)distance_hunter / max_d, 
		cast(f64)angle_target / 360.0, 
		cast(f64)angle_hunter / 360.0
	}

	dW1 : [N_neurons]f64
	dW2 : [N_outputs]f64
	dB1 : [N_neurons]f64
	dB2 : [N_outputs]f64
	
	A : [N_neurons]f64
	Y : [N_outputs]f64

	// Forward
	A = relu(W1 * math.sum(X[:]) + B1)
	Y = softmax(W2 * math.sum(A[:]) + B2)
	
	Loss := CrossEntropyLoss(Y[best_action_idx])

	// Backward
	sum_A := math.sum(A[:])
	sum_X := math.sum(X[:])
	sum_E2 : f64 = 0.0

	for i in 0 ..< len(W2) {
		sum_E2 += math.exp_f64(W2[i] * sum_A + B2[i]) + epsylon
	}

	for i in 0 ..< N_outputs {
		if i == best_action_idx {
			dW2[i] = -1.0 / Y[best_action_idx]
			dW2[i] *= sum_A * math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]) * sum_E2
			dW2[i] -= sum_A * math.pow(math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]), 2)
			dW2[i] /= math.pow(sum_E2, 2)

			dB2[i] = -1.0 / Y[best_action_idx]
			dB2[i] *= math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]) * sum_E2
			dB2[i] -= math.pow(math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]), 2)
			dB2[i] /= math.pow(sum_E2, 2)
		}
		else {
			dW2[i] = -1.0 / Y[best_action_idx]
			dW2[i] *= (-sum_A * math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]) * math.exp_f64(W2[i] * sum_A + B2[i]))
			dW2[i] /= math.pow(sum_E2, 2)

			dB2[i] = -1.0 / Y[best_action_idx]
			dB2[i] *= (-math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]) * math.exp_f64(W2[i] * sum_A + B2[i]))
			dB2[i] /= math.pow(sum_E2, 2)
		}
	}

	for i in 0 ..< N_neurons {
		if A[i] != 0 {
			dW1[i] = -1.0 / Y[best_action_idx]
			dW1[i] *= W2[best_action_idx] * math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]) * sum_E2
			dW1[i] /= math.pow(sum_E2, 2)
			dW1[i] *= W1[i] * sum_X + B1[i]

			dB1[i] = -1.0 / Y[best_action_idx]
			dB1[i] *= W2[best_action_idx] * math.exp_f64(W2[best_action_idx] * sum_A + B2[best_action_idx]) * sum_E2
			dB1[i] /= math.pow(sum_E2, 2)
		}
		else {
			dW1[i] = 0.0

			dB1[i] = 0.0
		}
	}

	// Update
	W1 -= alpha * dW1
	B1 -= alpha * dB1

	W2 -= alpha * dW2
	B2 -= alpha * dB2

	return Loss
}

NN_move :: proc(y, x, y_target, x_target, y_hunter, x_hunter : int) -> (step_y, step_x : int) {
	angle_target := math.atan2_f32(cast(f32)(y_target - y), cast(f32)(x_target - x)) * (180 / math.PI )
	angle_hunter := math.atan2_f32(cast(f32)(y_hunter - y), cast(f32)(x_hunter - x)) * (180 / math.PI )
	
	distance_target := math.sqrt_f32(cast(f32)((y_target - y)*(y_target - y) + (x_target - x)*(x_target - x)))
	distance_hunter := math.sqrt_f32(cast(f32)((y_hunter - y)*(y_hunter - y) + (x_hunter - x)*(x_hunter - x)))
	
	max_d := math.sqrt_f64(cast(f64)(math.pow_f64((HEIGHT-2), 2) + math.pow_f64((WIDTH-2), 2)))
	
	X := [4]f64 {
		cast(f64)distance_target / max_d, 
		cast(f64)distance_hunter / max_d, 
		cast(f64)angle_target / 360.0, 
		cast(f64)angle_hunter / 360.0
	}

	A : [N_neurons]f64
	Y : [N_outputs]f64
	
	A = relu(W1 * math.sum(X[:]) + B1)
	Y = softmax(W2 * math.sum(A[:]) + B2)

	max_Y := 0.0
	max_Y_idx := 0
	
	neighbors : [8]string
	neighbors[0] = FIELD[y-1][x]
	neighbors[1] = FIELD[y-1][x+1]
	neighbors[2] = FIELD[y][x+1]
	neighbors[3] = FIELD[y+1][x+1]
	neighbors[4] = FIELD[y+1][x]
	neighbors[5] = FIELD[y+1][x-1]
	neighbors[6] = FIELD[y][x-1]
	neighbors[7] = FIELD[y-1][x-1]

	for i in 0 ..< len(Y) {
		if max_Y < Y[i] && i == 0 {
			max_Y = Y[i]
			max_Y_idx = i
		}
		else if max_Y < Y[i] && neighbors[i-1] == "·" {
			max_Y = Y[i]
			max_Y_idx = i
		}
	}

	if max_Y_idx == 0 {
		step_y = 0
		step_x = 0
	}
	else if max_Y_idx == 1 {
		step_y = -1
		step_x = 0
	}
	else if max_Y_idx == 2 {
		step_y = -1
		step_x = 1
	}
	else if max_Y_idx == 3 {
		step_y = 0
		step_x = 1
	}
	else if max_Y_idx == 4 {
		step_y = 1
		step_x = 1
	}
	else if max_Y_idx == 5 {
		step_y = 1
		step_x = 0
	}
	else if max_Y_idx == 6 {
		step_y = 1
		step_x = -1	
	}
	else if max_Y_idx == 7 {
		step_y = 0
		step_x = -1
	}
	else if max_Y_idx == 8 {
		step_y = -1
		step_x = -1
	}

	fmt.println(step_y, step_x)

	return
}

move :: proc(y, x, y_target, x_target, y_hunter, x_hunter : int, symbol : string) -> (int, int) {
	min_d : f32 = 100_000.0
	step_y, step_x : int
	
	if FIELD[y-1][x] == "·" {
		d := math.sqrt_f32(cast(f32)((y-1-y_target)*(y-1-y_target) + (x-x_target)*(x-x_target)))
		if min_d > d {
			min_d = d
			step_y = -1
			step_x = 0
		}
	}

	if FIELD[y-1][x+1] == "·" {
		d := math.sqrt_f32(cast(f32)((y-1-y_target)*(y-1-y_target) + (x+1-x_target)*(x+1-x_target)))
		if min_d > d {
			min_d = d
			step_y = -1
			step_x = 1
		}
	}

	if FIELD[y][x+1] == "·" {
		d := math.sqrt_f32(cast(f32)((y-y_target)*(y-y_target) + (x+1-x_target)*(x+1-x_target)))
		if min_d > d {
			min_d = d
			step_y = 0
			step_x = 1
		}
	}

	if FIELD[y+1][x+1] == "·" {
		d := math.sqrt_f32(cast(f32)((y+1-y_target)*(y+1-y_target) + (x+1-x_target)*(x+1-x_target)))
		if min_d > d {
			min_d = d
			step_y = 1
			step_x = 1
		}
	}

	if FIELD[y+1][x] == "·" {
		d := math.sqrt_f32(cast(f32)((y+1-y_target)*(y+1-y_target) + (x-x_target)*(x-x_target)))
		if min_d > d {
			min_d = d
			step_y = 1
			step_x = 0
		}
	}

	if FIELD[y+1][x-1] == "·" {
		d := math.sqrt_f32(cast(f32)((y+1-y_target)*(y+1-y_target) + (x-1-x_target)*(x-1-x_target)))
		if min_d > d {
			min_d = d
			step_y = 1
			step_x = -1
		}
	}

	if FIELD[y][x-1] == "·" {
		d := math.sqrt_f32(cast(f32)((y-y_target)*(y-y_target) + (x-1-x_target)*(x-1-x_target)))
		if min_d > d {
			min_d = d
			step_y = 0
			step_x = -1
		}
	}

	if FIELD[y-1][x-1] == "·" {
		d := math.sqrt_f32(cast(f32)((y-1-y_target)*(y-1-y_target) + (x-1-x_target)*(x-1-x_target)))
		if min_d > d {
			min_d = d
			step_y = -1
			step_x = -1
		}
	}

	d_hunter := math.sqrt_f32(cast(f32)((y-y_hunter)*(y-y_hunter) + (x-x_hunter)*(x-x_hunter)))

	if d_hunter < min_d {
		max_d : f32 = 0.0

		if FIELD[y-1][x] == "·" {
			d := math.sqrt_f32(cast(f32)((y-1-y_hunter)*(y-1-y_hunter) + (x-x_hunter)*(x-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = -1
				step_x = 0
			}
		}

		if FIELD[y-1][x+1] == "·" {
			d := math.sqrt_f32(cast(f32)((y-1-y_hunter)*(y-1-y_hunter) + (x+1-x_hunter)*(x+1-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = -1
				step_x = 1
			}
		}

		if FIELD[y][x+1] == "·" {
			d := math.sqrt_f32(cast(f32)((y-y_hunter)*(y-y_hunter) + (x+1-x_hunter)*(x+1-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = 0
				step_x = 1
			}
		}

		if FIELD[y+1][x+1] == "·" {
			d := math.sqrt_f32(cast(f32)((y+1-y_hunter)*(y+1-y_hunter) + (x+1-x_hunter)*(x+1-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = 1
				step_x = 1
			}
		}

		if FIELD[y+1][x] == "·" {
			d := math.sqrt_f32(cast(f32)((y+1-y_hunter)*(y+1-y_hunter) + (x-x_hunter)*(x-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = 1
				step_x = 0
			}
		}

		if FIELD[y+1][x-1] == "·" {
			d := math.sqrt_f32(cast(f32)((y+1-y_hunter)*(y+1-y_hunter) + (x-1-x_hunter)*(x-1-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = 1
				step_x = -1
			}
		}
	
		if FIELD[y][x-1] == "·" {
			d := math.sqrt_f32(cast(f32)((y-y_hunter)*(y-y_hunter) + (x-1-x_hunter)*(x-1-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = 0
				step_x = -1
			}
		}
	
		if FIELD[y-1][x-1] == "·" {
			d := math.sqrt_f32(cast(f32)((y-1-y_hunter)*(y-1-y_hunter) + (x-1-x_hunter)*(x-1-x_hunter)))
			if max_d < d {
				max_d = d
				step_y = -1
				step_x = -1
			}
		}
	
	}
			
	return step_y, step_x

}

convert :: proc(y, x, y_target, x_target : int) {
	symbol := FIELD[y][x]
	enemy_symbol := FIELD[y_target][x_target]
	type : string
	SPRIdx, SPREnemyIdx : int

	if symbol == "R"{
		type = "Rock"
		SPRIdx = 2
		SPREnemyIdx = 0
	}
	else if symbol == "S" {
		type = "Scissor"
		SPRIdx = 0
		SPREnemyIdx = 1
	}
	else if symbol == "P" {
		type = "Paper"
		SPRIdx = 1
		SPREnemyIdx = 2
	}

	if y-1 == y_target && x == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y-1 == y_target && x+1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y == y_target && x+1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y+1 == y_target && x+1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y+1 == y_target && x == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y+1 == y_target && x-1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y == y_target && x-1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}

	if y-1 == y_target && x-1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol
			}
		}
	}
}

get_best_action :: proc(y, x, y_target, x_target, y_enemy, x_enemy : int) -> (action : int) {
	awards : [9]int = 0

	d_to_enemy := math.sqrt_f32(math.pow_f32(cast(f32)(y-y_enemy), 2) + math.pow_f32(cast(f32)(x-x_enemy), 2))
	d_to_target := math.sqrt_f32(math.pow_f32(cast(f32)(y-y_target), 2) + math.pow_f32(cast(f32)(x-x_target), 2))
	
	points := [4]int {1, 2, 1, 2}

	if FIELD[y-1][x] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y-1-y_enemy)*(y-1-y_enemy) + (x-x_enemy)*(x-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y-1-y_target)*(y-1-y_target) + (x-x_target)*(x-x_target)))

		if d_e > d_to_enemy {
			awards[1] += points[0]
		}
		else {
			awards[1] -= points[1]
		}

		if d_t < d_to_target {
			awards[1] += points[2]
		}
		else {
			awards[1] -= points[3]
		}
	}

	if FIELD[y-1][x+1] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y-1-y_enemy)*(y-1-y_enemy) + (x+1-x_enemy)*(x+1-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y-1-y_target)*(y-1-y_target) + (x+1-x_target)*(x+1-x_target)))

		if d_e > d_to_enemy {
			awards[2] += points[0]
		}
		else {
			awards[2] -= points[1]
		}

		if d_t < d_to_target {
			awards[2] += points[2]
		}
		else {
			awards[2] -= points[3]
		}
	}

	if FIELD[y][x+1] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y-y_enemy)*(y-y_enemy) + (x+1-x_enemy)*(x+1-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y-y_target)*(y-y_target) + (x+1-x_target)*(x+1-x_target)))

		if d_e > d_to_enemy {
			awards[3] += points[0]
		}
		else {
			awards[3] -= points[1]
		}

		if d_t < d_to_target {
			awards[3] += points[2]
		}
		else {
			awards[3] -= points[3]
		}
	}

	if FIELD[y+1][x+1] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y+1-y_enemy)*(y+1-y_enemy) + (x+1-x_enemy)*(x+1-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y+1-y_target)*(y+1-y_target) + (x+1-x_target)*(x+1-x_target)))

		if d_e > d_to_enemy {
			awards[4] += points[0]
		}
		else {
			awards[4] -= points[1]
		}

		if d_t < d_to_target {
			awards[4] += points[2]
		}
		else {
			awards[4] -= points[3]
		}
	}

	if FIELD[y+1][x] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y+1-y_enemy)*(y+1-y_enemy) + (x-x_enemy)*(x-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y+1-y_target)*(y+1-y_target) + (x-x_target)*(x-x_target)))

		if d_e > d_to_enemy {
			awards[5] += points[0]
		}
		else {
			awards[5] -= points[1]
		}

		if d_t < d_to_target {
			awards[5] += points[2]
		}
		else {
			awards[5] -= points[3]
		}
	}

	if FIELD[y+1][x-1] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y+1-y_enemy)*(y-1-y_enemy) + (x-1-x_enemy)*(x-1-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y+1-y_target)*(y-1-y_target) + (x-1-x_target)*(x-1-x_target)))

		if d_e > d_to_enemy {
			awards[6] += points[0]
		}
		else {
			awards[6] -= points[1]
		}

		if d_t < d_to_target {
			awards[6] += points[2]
		}
		else {
			awards[6] -= points[3]
		}
	}

	if FIELD[y][x-1] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y-y_enemy)*(y-y_enemy) + (x-1-x_enemy)*(x-1-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y-y_target)*(y-y_target) + (x-1-x_target)*(x-1-x_target)))

		if d_e > d_to_enemy {
			awards[7] += points[0]
		}
		else {
			awards[7] -= points[1]
		}

		if d_t < d_to_target {
			awards[7] += points[2]
		}
		else {
			awards[7] -= points[3]
		}
	}

	if FIELD[y-1][x-1] == "·" {
		d_e := math.sqrt_f32(cast(f32)((y-1-y_enemy)*(y-1-y_enemy) + (x-1-x_enemy)*(x-1-x_enemy)))
		d_t := math.sqrt_f32(cast(f32)((y-1-y_target)*(y-1-y_target) + (x-1-x_target)*(x-1-x_target)))

		if d_e > d_to_enemy {
			awards[8] += points[0]
		}
		else {
			awards[8] -= points[1]
		}

		if d_t < d_to_target {
			awards[8] += points[2]
		}
		else {
			awards[8] -= points[3]
		}
	}
	
	max_award := awards[1]
	action = 0

	for i in 0 ..< len(awards) {
		if max_award < awards[i] {
			max_award = awards[i]
			action = i
		}
	}

	return 
}

HEIGHT :: 40
WIDTH :: 60
FIELD : [HEIGHT][WIDTH]string

N :: 10
Entities : [N*3]Entity

SPRCount : [3]int

Scissor :: "S"
Paper :: "P"
Rock :: "R"

N_neurons :: 4
N_outputs :: 9
N_epochs :: 100

W1 : [N_neurons]f64 
W2 : [N_outputs]f64 
B1 : [N_neurons]f64
B2 : [N_outputs]f64

main :: proc() {
	types := [3]string {"Scissor", "Paper", "Rock"}
	symbols := [3]string {Scissor, Paper, Rock}

	for i in 0 ..< N_neurons {
		W1[i] = rand.norm_float64()
		B1[i] = rand.norm_float64()

		if W1[i] < 0 { W1[i] *= -1 }
		if B1[i] < 0 { B1[i] *= -1 }
	}

	for i in 0 ..< N_outputs {
		W2[i] = rand.norm_float64()
		B2[i] = rand.norm_float64()
		
		if W2[i] < 0 { W2[i] *= -1 }
		if B2[i] < 0 { B2[i] *= -1 }
	}

	for epoch in 0 ..< N_epochs {
		stop_flag := false
		winner := 0

		for i in 0 ..< 3 {
			for j in 0 ..< N {
				x := rand.int31_max(WIDTH-6) + 3
				y := rand.int31_max(HEIGHT-6) + 3
				Entities[i*N+j] = {types[i], symbols[i], cast(int)(y), cast(int)(x)}
			}
		}
	
		_ = reset_field()

		for {
			time.sleep(100 * time.Millisecond)
			fmt.print("\x1B[2J\x1B[H")

			print_field()
			Entities_copy := Entities
		
			avg_loss := 0.0
			
			for idx in 0 ..< len(Entities) {
				enemy_y, enemy_x := closest_enemy(Entities[idx])
				hunter_y, hunter_x := closest_hunter(Entities[idx])
				
				step_y, step_x := move(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_x, Entities[idx].symbol)

				if Entities[idx].type == "Rock" {
					angle_target := math.atan2_f32(cast(f32)(enemy_y - Entities[idx].y), cast(f32)(enemy_x - Entities[idx].x)) * (180.0/math.PI)
					angle_hunter := math.atan2_f32(cast(f32)(hunter_y - Entities[idx].y), cast(f32)(hunter_x - Entities[idx].x)) * (180.0/math.PI)

					distance_target := math.sqrt_f32((math.pow_f32(cast(f32)(enemy_y - Entities[idx].y), 2) + math.pow(cast(f32)(enemy_x - Entities[idx].x), 2)))
					distance_hunter := math.sqrt_f32((math.pow_f32(cast(f32)(hunter_y - Entities[idx].y), 2) + math.pow(cast(f32)(hunter_x - Entities[idx].x), 2)))

					best_action := get_best_action(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_x)	

					avg_loss += NN_train(distance_target, distance_hunter, angle_target, angle_hunter, best_action, avg_loss)
					
					step_y, step_x = NN_move(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_y)
				}
				//else {
				//	step_y, step_x = move(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_x, Entities[idx].symbol)
				//}

				Entities[idx].y += step_y 
				Entities[idx].x += step_x
			
				_ = reset_field()

				convert(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x)

				SPRCount = reset_field()

				if SPRCount[0] == 0 {
					stop_flag = true
					winner = 1
					break
				}
				else if SPRCount[1] == 0 {
					stop_flag = true
					winner = 2
				}
				else if SPRCount[2] == 0 {
					stop_flag = true
					winner = 0
				}
			}
			
			if stop_flag {
				break
			}

			fmt.println("\nEpoch:", epoch)
			fmt.println("Avg Loss:", avg_loss / cast(f64)SPRCount[2])
			fmt.println("Scissors:", SPRCount[0])
			fmt.println("Papers:", SPRCount[1])
			fmt.println("Rocks:", SPRCount[2])

			if SPRCount[0] == N*3 {
				fmt.print("\x1B[2J\x1B[H")
				print_field()
				fmt.println("Scissors win!")
				break
			}
			else if SPRCount[1] == N*3 {
				fmt.print("\x1B[2J\x1B[H")
				print_field()
				fmt.println("Papers win!")
				break
			}
			else if SPRCount[2] == N*3 {
				fmt.print("\x1B[2J\x1B[H")
				print_field()
				fmt.println("Rocks win!")
				break
			}
		}
	}
	
	_ = reset_field()
	
	for epoch in 0 ..< N_epochs {
		stop_flag := false
		winner := 0

		for i in 0 ..< 3 {
			for j in 0 ..< N {
				x := rand.int31_max(WIDTH-6) + 3
				y := rand.int31_max(HEIGHT-6) + 3
				Entities[i*N+j] = {types[i], symbols[i], cast(int)(y), cast(int)(x)}
			}
		}

		for {
			time.sleep(200 * time.Millisecond)
			fmt.print("\x1B[2J\x1B[H")
	
			print_field()
			
			for idx in 0 ..< len(Entities) {
				enemy_y, enemy_x := closest_enemy(Entities[idx])
				hunter_y, hunter_x := closest_hunter(Entities[idx])
				
				step_y, step_x := 0, 0
	
				//if Entities[idx].type == "Rock" {
					step_y, step_x = NN_move(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_y)
				//}
				//else {
				//	step_y, step_x = move(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_x, Entities[idx].symbol)
				//}

				Entities[idx].y += step_y 
				Entities[idx].x += step_x
				
				_ = reset_field()
	
				convert(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x)
	
				SPRCount = reset_field()
	
				if SPRCount[0] == 0 {
					stop_flag = true
					winner = 1
					break
				}
				else if SPRCount[1] == 0 {
					stop_flag = true
					winner = 2
				}
				else if SPRCount[2] == 0 {
					stop_flag = true
					winner = 0
				}
			}
			
			if stop_flag {
				fmt.println("\nEpoch:", epoch)
				fmt.println("Scissors:", SPRCount[0])
				fmt.println("Papers:", SPRCount[1])
				fmt.println("Rocks:", SPRCount[2])
		
				if winner == 0 {
					fmt.print("\x1B[2J\x1B[H")
					print_field()
					fmt.println("Scissors win!")
					break
				}
				else if winner == 1 {
					fmt.print("\x1B[2J\x1B[H")
					print_field()
					fmt.println("Papers win!")
					break
				}
				else if winner == 2 {
					fmt.print("\x1B[2J\x1B[H")
					print_field()
					fmt.println("Rocks win!")
					break
				}
			}

			fmt.println("Scissors:", SPRCount[0])
			fmt.println("Papers:", SPRCount[1])
			fmt.println("Rocks:", SPRCount[2])
		}
	}

	fmt.println(W1)
	fmt.println(B1)
	fmt.println(W2)
	fmt.println(B2)
}
