package main

import "core:fmt"
import "core:math"
import "core:time"
import "core:math/rand"

Directions :: enum { N, NE, E, SE, S, SW, W, NW }

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

reset_field :: proc() {
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
	}
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

closest_enemy :: proc(e : Entity) -> (int, int) {
	enemy_type, hunter_type := enemy_type(e.type)
	min_d : f32 = 100_000.0
	min_y, min_x : int

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

	return min_y, min_x
}

closest_hunter :: proc(e : Entity) -> (int, int) {
	enemy_type, hunter_type := enemy_type(e.type)
	min_d : f32 = 100_000.0
	min_y, min_x : int

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

	return min_y, min_x
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

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y-1 == y_target && x+1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y == y_target && x+1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y+1 == y_target && x+1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y+1 == y_target && x == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y+1 == y_target && x-1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y == y_target && x-1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}

	if y-1 == y_target && x-1 == x_target {
		for idx in 0 ..< len(Entities) {
			if Entities[idx].y == y_target && Entities[idx].x == x_target {
				Entities[idx].type = type
				Entities[idx].symbol = symbol

				SPRCount[SPRIdx] += 1
				SPRCount[SPREnemyIdx] -= 1
			}
		}
	}
}

HEIGHT :: 40
WIDTH :: 80
FIELD : [HEIGHT][WIDTH]string

N :: 10
Entities : [N*3]Entity

SPRCount := [3]int {N, N, N}

Scissor :: "S"
Paper :: "P"
Rock :: "R"

main :: proc() {
	types := [3]string {"Scissor", "Paper", "Rock"}
	symbols := [3]string {"S", "P", "R"}
	
	for i in 0 ..< 3 {
		for j in 0 ..< N {
			x := rand.int31_max(WIDTH-6) + 3
			y := rand.int31_max(HEIGHT-6) + 3
			Entities[i*N+j] = {types[i], symbols[i], cast(int)(y), cast(int)(x)}
		}
	}
	
	reset_field()

	for {
		time.sleep(300 * time.Millisecond)
		fmt.print("\x1B[2J\x1B[H")

		print_field()
		
		for idx in 0 ..< len(Entities) {
			enemy_y, enemy_x := closest_enemy(Entities[idx])
			hunter_y, hunter_x := closest_hunter(Entities[idx])
			step_y, step_x := move(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x, hunter_y, hunter_x, Entities[idx].symbol)

			Entities[idx].y += step_y 
			Entities[idx].x += step_x
			
			reset_field()

			convert(Entities[idx].y, Entities[idx].x, enemy_y, enemy_x)

			reset_field()
		}
		fmt.println("Scissors:", SPRCount[0])
		fmt.println("Papers:", SPRCount[1])
		fmt.println("Rocks:", SPRCount[2])

		if SPRCount[0] == N*3 {
			print_field()
			fmt.println("Scissors win!")
			break
		}
		else if SPRCount[1] == N*3 {
			print_field()
			fmt.println("Papers win!")
			break
		}
		else if SPRCount[2] == N*3 {
			print_field()
			fmt.println("Rocks win!")
			break
		}
	}
}
