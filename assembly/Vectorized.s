# define STDOUT 0xd0580000

.section .text
.global _start
## START YOUR CODE HERE
_start:


    # === Normalize image ===
##################################################################################
#This part takes raw image pixels (0-255) and turns them into floating point. 
#We do this because when training it will be easier for neural nets to work on normalized values. .bss
###################################################################################

# .text
# .globl main

main:
    # Save return address
    addi sp, sp, -4
    sw ra, 0(sp)

    # Load addresses
    la a1, input_matrix      # Input matrix (same for all calls)
    la a2, conv_output       # Output matrix (will append results)
    
    # Load filter and bias base addresses
    la a0, conv_filters           # Filter weights
    la a4, filter_bias       # Bias values
    
    # Constants
    li s0, 8                 # Number of filters
    li s1, 0                 # Filter counter
    
filter_loop:
    bge s1, s0, end_loop     # Process all 8 filters
    
    # Calculate current filter and bias addresses
    li t0, 100               # Size of one filter (25 elements * 4 bytes)
    mul t1, s1, t0           # Filter offset
    add a0, a0, t1           # Current filter address
    
    slli t1, s1, 2           # Bias offset (4 bytes per bias)
    add a4, a4, t1           # Current bias address
    
    # Call convolution function
    jal conv2d
    
    # Update output pointer (24x24 elements per filter)
    li t0, 2304              # 24*24*4 bytes per output
    add a2, a2, t0
    
    # Next filter
    addi s1, s1, 1
    j filter_loop

end_loop:
    # Restore return address
    lw ra, 0(sp)
    addi sp, sp, 4
    ret

# Convolution function (optimized version from previous answer)
conv2d:
    # Save registers
    addi sp, sp, -32
    sw s0, 0(sp)
    sw s1, 4(sp)
    sw s2, 8(sp)
    sw s3, 12(sp)
    sw s4, 16(sp)
    sw s5, 20(sp)
    sw s6, 24(sp)
    sw s7, 28(sp)

    # Load bias value
    flw f18, 0(a4)
    
    # Constants
    li t0, 24        # output dimension (28-5+1)
    li t1, 112       # input row stride (28*4)
    li t2, 20        # filter row size (5*4)
    li t3, 96        # output row stride (24*4)
    
    # Initialize vertical position counter
    li s9, 0
    
vertical_loop:
    bge s9, t0, conv_exit
    
    # Calculate input row pointer
    mul t4, s9, t1
    add t4, a1, t4   # input row pointer
    
    # Initialize horizontal position counter
    li s7, 0
    
horizontal_loop:
    bge s7, t0, next_vertical
    
    # Initialize accumulator with bias
    fmv.s f17, f18
    
    # ===== Row 0 =====
    # Load filter row 0
    flw f1, 0(a0)    # filter[0][0]
    flw f2, 4(a0)    # filter[0][1]
    flw f3, 8(a0)    # filter[0][2]
    flw f4, 12(a0)   # filter[0][3]
    flw f5, 16(a0)   # filter[0][4]
    
    # Load input row 0
    flw f6, 0(t4)    # input[0][0]
    flw f7, 4(t4)    # input[0][1]
    flw f8, 8(t4)    # input[0][2]
    flw f9, 12(t4)   # input[0][3]
    flw f10, 16(t4)  # input[0][4]
    
    # Compute dot product
    fmul.s f16, f1, f6
    fmadd.s f16, f2, f7, f16
    fmadd.s f16, f3, f8, f16
    fmadd.s f16, f4, f9, f16
    fmadd.s f16, f5, f10, f16
    fadd.s f17, f17, f16
    
    # ===== Row 1 =====
    # Calculate pointers for row 1
    add t5, t4, t1   # input row 1
    add t6, a0, t2   # filter row 1
    
    # Load filter row 1
    flw f1, 0(t6)
    flw f2, 4(t6)
    flw f3, 8(t6)
    flw f4, 12(t6)
    flw f5, 16(t6)
    
    # Load input row 1
    flw f6, 0(t5)
    flw f7, 4(t5)
    flw f8, 8(t5)
    flw f9, 12(t5)
    flw f10, 16(t5)
    
    # Compute dot product
    fmul.s f16, f1, f6
    fmadd.s f16, f2, f7, f16
    fmadd.s f16, f3, f8, f16
    fmadd.s f16, f4, f9, f16
    fmadd.s f16, f5, f10, f16
    fadd.s f17, f17, f16
    
    # ===== Row 2 =====
    # Calculate pointers for row 2
    add t5, t5, t1   # input row 2
    add t6, t6, t2   # filter row 2
    
    # Load filter row 2
    flw f1, 0(t6)
    flw f2, 4(t6)
    flw f3, 8(t6)
    flw f4, 12(t6)
    flw f5, 16(t6)
    
    # Load input row 2
    flw f6, 0(t5)
    flw f7, 4(t5)
    flw f8, 8(t5)
    flw f9, 12(t5)
    flw f10, 16(t5)
    
    # Compute dot product
    fmul.s f16, f1, f6
    fmadd.s f16, f2, f7, f16
    fmadd.s f16, f3, f8, f16
    fmadd.s f16, f4, f9, f16
    fmadd.s f16, f5, f10, f16
    fadd.s f17, f17, f16
    
    # ===== Row 3 =====
    # Calculate pointers for row 3
    add t5, t5, t1   # input row 3
    add t6, t6, t2   # filter row 3
    
    # Load filter row 3
    flw f1, 0(t6)
    flw f2, 4(t6)
    flw f3, 8(t6)
    flw f4, 12(t6)
    flw f5, 16(t6)
    
    # Load input row 3
    flw f6, 0(t5)
    flw f7, 4(t5)
    flw f8, 8(t5)
    flw f9, 12(t5)
    flw f10, 16(t5)
    
    # Compute dot product
    fmul.s f16, f1, f6
    fmadd.s f16, f2, f7, f16
    fmadd.s f16, f3, f8, f16
    fmadd.s f16, f4, f9, f16
    fmadd.s f16, f5, f10, f16
    fadd.s f17, f17, f16
    
    # ===== Row 4 =====
    # Calculate pointers for row 4
    add t5, t5, t1   # input row 4
    add t6, t6, t2   # filter row 4
    
    # Load filter row 4
    flw f1, 0(t6)
    flw f2, 4(t6)
    flw f3, 8(t6)
    flw f4, 12(t6)
    flw f5, 16(t6)
    
    # Load input row 4
    flw f6, 0(t5)
    flw f7, 4(t5)
    flw f8, 8(t5)
    flw f9, 12(t5)
    flw f10, 16(t5)
    
    # Compute dot product
    fmul.s f16, f1, f6
    fmadd.s f16, f2, f7, f16
    fmadd.s f16, f3, f8, f16
    fmadd.s f16, f4, f9, f16
    fmadd.s f16, f5, f10, f16
    fadd.s f17, f17, f16
    
    # ===== Store Result =====
    # Calculate output position
    mul t5, s9, t3   # vertical offset
    slli t6, s7, 2   # horizontal offset
    add t5, t5, t6   # total offset
    add t5, a2, t5   # output pointer
    
    # Store result
    fsw f17, 0(t5)
    
    # Next horizontal position
    addi s7, s7, 1
    addi t4, t4, 4   # move input pointer right
    j horizontal_loop

next_vertical:
    addi s9, s9, 1
    j vertical_loop

conv_exit:
    # Restore registers
    lw s0, 0(sp)
    lw s1, 4(sp)
    lw s2, 8(sp)
    lw s3, 12(sp)
    lw s4, 16(sp)
    lw s5, 20(sp)
    lw s6, 24(sp)
    lw s7, 28(sp)
    addi sp, sp, 32
    
    #Since a2 was incremented after each filter, rewind by 2304 to point to the last output
    li t0, 2304
    sub a0, a2, t0      # Point back to last filter output
    # li a1, 24           # Since 24x24 output
    # call printToLogVectorized
    # j _finish


# # =============================================================================
# # Linking Conv Layer with ReLU
# # Done by SafeGOAT
# # Scroll Down to see ReLU subroutine
# # =============================================================================
la a0, conv_output # Input to ReLU is the output of convolutional layer
li a1, 4608 # size of the output of the convolutional layer 24x24x8
addi sp, sp, -16    # Pre-align for ReLU's stack usage 
call relu_activation # Call ReLU subroutine
addi sp, sp, 16     # Rebalance stack  
# li a1, 24           # Since 24x24 output
# call printToLogVectorized
# j _finish
# # ReLU done
# # =============================================================================

call maxpool_2x2

maxpool_2x2:
    mv t0, a0
    la t1, output_max
    li a2, 0             # this will keep incrementing by 4 to store at next position in the output_max_matrix
    
    li s1, 96
    li s2, 0             # block index i (outer loop)


    li t4, 16            # row stride in bytes (4 floats * 4 bytes) (stride = 2)
    li t5, 0             # temp offset for current patch 
 


max_pool_outer_loop:
    bge s2, s1, done    # done label has to be replaced with the label of the next layer/function in our cnn
    addi s4, zero, 192
    mul s5, s4, s2
    addi s2, s2, 1
    li t2, 12            # number of patches to process (12 here) --> We are moving horizontally
    li t3, 0             # patch index j (inner loop)
    j patch_loop



patch_loop:
    bge t3, t2, max_pool_outer_loop     # exit if all patches are processed

    # Calculate column offset: col = (i % 2) * 8
    # For generality, though here it's just col = i * 8
    addi a1 , zero, 8
    mul t5, t3, a1        # column offset for each patch in bytes done

    # Row 1, Col 1 → offset 0 + t5
    add t5, t5, s5  # from outer loop
    add t6, t0, t5
    flw f1, 0(t6)

    # Row 1, Col 2 → offset 4 + t5
    flw f2, 4(t6)

    # Row 2, Col 1 → offset 16 + t5
    flw f3, 96(t6)

    # Row 2, Col 2 → offset 20 + t5
    flw f4, 100(t6)

    # Compute max of the four
    fmax.s f1, f1, f2
    fmax.s f1, f1, f3
    fmax.s f1, f1, f4

    
    # Store the result in output_max[i]
    add a4, t1, a2
    fsw f1, 0(a4)
    addi a2, a2, 4

    addi t3, t3, 1       # i++ (increment the inner loop)
    j patch_loop

done:
    call dense_layer
    # mv a0, t1
    # li a1, 12
    # call printToLogVectorized
    # j _finish
    


# # =============================================================================
# # ReLU Activation Function (In-Place Operation)
# #
# # Applies the Rectified Linear Unit (ReLU) activation function to a float32 matrix
# # in memory. ReLU is defined as: f(x) = max(0, x)
# #
# # Arguments:
# #   a0 - Pointer to the start of the matrix data (must be word-aligned)
# #   a1 - Number of elements in the matrix (must be > 0)
# #
# # Supported Matrix Formats:
# #   - Contiguous float32 values in memory (4 bytes per element)
# #   - Row-major or column-major order (handled equivalently)
# #
# # Register Usage:
# #   Preserved: s0, fs0, ra (all other registers are caller-saved)
# #   Temp: t0, ft0, ft1
# #
# # Stack Usage:
# #   - Allocates 16 bytes in prologue (for saved registers)
# #   - Temporarily uses 4 bytes to load 0.0
# #   - Maintains 16-byte stack alignment
# #
# # Performance:
# #   - Processes elements sequentially
# #   - Loads 0.0 constant once outside loop
# #   - No heap allocations or system calls
# #
# # Example Usage:
# #   la a0, input_matrix    # Load matrix address
# #   li a1, 16             # 4x4 matrix elements
# #   call relu_activation  # Apply in-place
# #
# # Safety Notes:
# #   1. Matrix memory must be writable
# #   2. Count (a1) must match allocated memory
# #   3. Does not validate input alignment
# #   4. Preserves all registers except temp registers
# #
# # Error Handling:
# #   - No explicit error checking (maximizes performance)
# #   - Caller must ensure valid arguments
# #
# # Testing Recommendations:
# #   1. Test with mixed positive/negative/zero values
# #   2. Verify edge cases (-0.0, denormals)
# #   3. Check single-element and large matrices
# # =============================================================================
relu_activation:
    # PROLOGUE
    addi sp, sp, -16        # Allocate stack space
    sw ra, 12(sp)           # Save return address
    fsw fs0, 8(sp)          # Save preserved FP register
    sw s0, 4(sp)            # Save preserved integer register
    
    # Load 0.0 using stack space
    addi sp, sp, -4         # Temporary space for 0.0
    sw zero, 0(sp)
    flw ft0, 0(sp)          # ft0 = 0.0
    addi sp, sp, 4          # Deallocate temp space
    
    # Initialize counter
    li t0, 0                # t0 = counter
    mv s0, a0               # s0 = preserved matrix pointer
    fmv.s fs0, ft0          # fs0 = preserved 0.0

relu_loop:
    flw ft1, 0(s0)          # Load current element
    fmax.s ft1, ft1, fs0    # ReLU: max(x, 0)
    fsw ft1, 0(s0)          # Store result back
    addi s0, s0, 4          # Next element
    addi t0, t0, 1          # Increment counter
    blt t0, a1, relu_loop   # Loop if not done

    # EPILOGUE
    lw s0, 4(sp)            # Restore saved register
    flw fs0, 8(sp)          # Restore FP register
    lw ra, 12(sp)           # Restore return address
    addi sp, sp, 16         # Deallocate stack space
    ret                     # Return to caller



#DENSE LAYER----------------------------------------------------------------------------------------------------------------------------#

dense_layer:
    # Load base addresses
    la a0, output_max        # Input vector (3 elements)
    la a1, dense_weights     # Weights (3x10, ROW-MAJOR)
    la a2, dense_bias        # Bias (10 elements)
    la a3, dense_output      # Output (10 elements)

    li t0, 0                 # Output index i = 0
    li t1, 10                # Number of outputs (columns)

dense_outer_loop: 
    bge t0, t1, dense_done   # Exit if all outputs processed

    fmv.s.x f0, x0           # Initialize accumulator = 0.0
    li t3, 0                 # Input index j = 0
    li t4, 3                 # Number of inputs (rows) = 3

dense_inner_loop:
    bge t3, t4, dense_inner_done  # Exit after 3 inputs

    # Calculate weight offset: weights[j][i] (ROW-MAJOR)
    # Offset = (j * 10 + i) * 4
    li t6, 10                # 10 columns per row
    mul t5, t3, t6           # j * 10 (start of row j)
    add t5, t5, t0           # + i (column index)
    slli t5, t5, 2           # Convert to byte offset (*4)
    add t6, a1, t5           # t6 = &weights[j][i]
    flw ft1, 0(t6)           # ft1 = weights[j][i]

    # Load input[j]
    slli t6, t3, 2           # j * 4
    add t6, a0, t6           # t6 = &input[j]
    flw ft2, 0(t6)           # ft2 = input[j]

    # Multiply and accumulate
    fmul.s ft3, ft1, ft2     # ft3 = weights[j][i] * input[j]
    fadd.s f0, f0, ft3       # accumulator += ft3

    addi t3, t3, 1           # j++
    j dense_inner_loop

dense_inner_done:
    # Add bias[i]
    slli t6, t0, 2           # i * 4
    add t6, a2, t6           # t6 = &bias[i]
    flw ft4, 0(t6)           # ft4 = bias[i]
    fadd.s f0, f0, ft4       # accumulator += bias[i]

    # Store result[i]
    slli t6, t0, 2           # i * 4
    add t6, a3, t6           # t6 = &output[i]
    fsw f0, 0(t6)            # output[i] = accumulator

    addi t0, t0, 1           # i++
    j dense_outer_loop

dense_done:
    call softmax_layer

# dense_layer:
#     # Load base addresses
#     la a0, output_max        # Input vector base
#     la a1, dense_weights     # Weights base
#     la a2, dense_bias        # Bias base
#     la a3, dense_output      # Output base

#     li t0, 0                 # Output index i = 0

# dense_outer_loop:
#     li t1, 0                 # Input index j = 0
#     fmv.s.x f0, x0           # f0 = 0.0, accumulator for dot product

#     mv s0, a0                # s0 = input base (output_max)
#     mv s1, a1                # s1 = current weight row start
#     li t2, 1152              # input size

# dense_inner_loop:
#     beq t1, t2, dense_inner_done

#     # Load input[j] into ft1
#     slli t3, t1, 2           # offset = j * 4
#     add t4, s0, t3           # address = input_base + offset
#     flw ft1, 0(t4)           # ft1 = input[j]

#     # Load weight[i * 1152 + j] into ft2
#     add t5, s1, t3           # address = weight_row_base + offset
#     flw ft2, 0(t5)           # ft2 = weight[i][j]

#     # Multiply and accumulate: f0 += ft1 * ft2
#     fmul.s ft3, ft1, ft2
#     fadd.s f0, f0, ft3

#     addi t1, t1, 1
#     j dense_inner_loop

# dense_inner_done:
#     # Load bias[i] into ft4
#     slli t3, t0, 2
#     add t4, a2, t3
#     flw ft4, 0(t4)

#     # Add bias: f0 += ft4
#     fadd.s f0, f0, ft4

#     # Store result in dense_output[i]
#     add t5, a3, t3
#     fsw f0, 0(t5)

#     # Prepare for next output neuron
#     addi t0, t0, 1
#     li t6, 10
#     beq t0, t6, dense_done

#     # Move weight pointer to next row: a1 += 1152 * 4 = 4608
#     li t2, 4608
#     add a1, a1, t2
#     j dense_outer_loop

# dense_done:
#     # la a0, dense_output
#     # li a1, 10
#     # call printToLogVectorized
#     # j _finish
#     # ret
#     call softmax_layer


#####BEGINNING OF SOFTMAX WITH NORMALIZATION AND EXP(FINAL)##########################################################
softmax_layer:
    la a3, dense_output
    la a1, p
    # Find max(Z[i]) for stability
    li t0, 0                    # i = 0
    flw f0, 0(a3)               # f0 = max = Z[0]

max_loop:
    li t1, 10
    bge t0, t1, compute_exp_sum

    slli t2, t0, 2              # t2 = i * 4
    add t3, a3, t2              # t3 = address of Z[i]
    flw f1, 0(t3)               # f1 = Z[i]
    fmax.s f0, f0, f1           # f0 = max(max, Z[i])

    addi t0, t0, 1              # i++
    j max_loop

# ---------------------------------
# Step 2: Compute exp(Z[i] - max) ≈ 1 + x + 0.5x² and store in p[i]
# Also accumulate sum in f2
# ---------------------------------
compute_exp_sum:
    li t0, 0                    # i = 0
    fmv.s.x f2, zero            # f2 = sum = 0.0

exp_loop:
    li t1, 10
    bge t0, t1, normalize_loop  # If i >= 10, go to normalization

    slli t2, t0, 2              # t2 = i * 4
    add t3, a3, t2              # t3 = address of Z[i]
    flw f1, 0(t3)               # f1 = Z[i]
    fsub.s f3, f1, f0           # f3 = x = Z[i] - max

    # Approximate exp(x): f4 = 1 + x + 0.5 * x²
    li t4, 0x3f800000           # 1.0 (float)
    fmv.s.x f4, t4              # f4 = 1.0
    fadd.s f4, f4, f3           # f4 = 1 + x

    fmul.s f5, f3, f3           # f5 = x²
    li t5, 0x3f000000           # 0.5 (float)
    fmv.s.x f6, t5              # f6 = 0.5
    fmul.s f5, f5, f6           # f5 = 0.5 * x²

    fadd.s f4, f4, f5           # f4 = 1 + x + 0.5x²

    # Store exp_approx in p[i]
    add t3, a1, t2              # address of p[i]
    fsw f4, 0(t3)               # store p[i]

    # Add to sum
    fadd.s f2, f2, f4           # sum += exp_approx

    addi t0, t0, 1              # i++
    j exp_loop

# -------------------------------
# Step 3: Normalize: p[i] /= sum
# -------------------------------
normalize_loop:
    li t0, 0                    # i = 0

norm_loop:
    li t1, 10
    bge t0, t1, done_softmax    # If i >= 10, finish

    slli t2, t0, 2              # t2 = i * 4
    add t3, a1, t2              # t3 = address of p[i]
    flw f1, 0(t3)               # f1 = p[i]
    fdiv.s f1, f1, f2           # f1 = p[i] / sum
    fsw f1, 0(t3)               # store back p[i]

    addi t0, t0, 1              # i++
    j norm_loop

done_softmax:
    la a0, p
    li a1, 10
    call printToLogVectorized
    j _finish
    ret                         # Return to caller



# Function: print
# Logs values from array in a0 into registers v1 for debugging and output_max.
# Inputs:
#   - a0: Base address of array
#   - a1: Size of array i.e. number of elements to log ## Safeguard: Actually it represents the rows of a square matrix though we could remove mul a1, a1, a1 line to work with a 1D array
# Clobbers: t0,t1, t2,t3 ft0, ft1.
printToLogVectorized:        
    addi sp, sp, -4
    sw a0, 0(sp)

    li t0, 0x123                 # Pattern for help in python script
    li t0, 0x456                 # Pattern for help in python script
    mv a1, a1                   # moving size to get it from log 
    # mul a1, a1, a1              # sqaure matrix has n^2 elements 
	addi t0, x0, 0		                # load i = 0
    printloop:
        vsetvli t3, a1, e32           # Set VLEN based on a1
        slli t4, t3, 2                # Compute VLEN * 4 for address increment

        vle32.v v1, (a0)              # Load real[i] into v1
        add a0, a0, t4                # Increment pointer for real[] by VLEN * 4
        add t0, t0, t3                # Increment index

        bge t0, a1, endPrintLoop      # Exit loop if i >= size
        j printloop                   # Jump to start of loop
    endPrintLoop:
    li t0, 0x123                    # Pattern for help in python script
    li t0, 0x456                    # Pattern for help in python script
	
    lw a0, 0(sp)
    addi sp, sp, 4
	jr ra #(from TA's original code so commented out)
    # Important (creating infinite loop during runtime)

# Function: _finish
# VeeR Related function which writes to to_host which stops the simulator
_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish

    .rept 100
        nop
    .endr

## ALL DATA IS DEFINED HERE LIKE MATRIX, CONSTANTS ETC


# DATA DEFINE START
.equ MatrixSize, 10
matrix1:
    .float -111.75, -602.75, 646.0, -439.5, -25.75, 351.25, -699.0, 736.5, -69.5, -453.75
    .float -195.5, 254.5, -616.75, 712.5, 382.75, 532.25, 656.0, 309.0, -580.0, -91.75
    .float 793.25, -207.0, -893.25, -652.25, 66.25, -734.5, -522.0, 68.75, 894.75, -80.0
    .float 18.5, 953.5, 288.75, -235.0, 780.25, -577.0, -959.5, 723.25, -513.5, -909.75
    .float 310.0, 318.0, 973.75, 1.25, 443.5, 982.5, 265.75, -552.5, -273.0, 740.25
    .float -885.75, -964.5, 485.75, -134.75, -399.0, -374.0, 205.5, -62.75, 3.75, 442.5
    .float -23.25, 182.0, 845.0, -370.75, 450.5, -932.25, 779.0, -635.75, 571.75, 102.75
    .float 137.5, 568.25, -114.0, 813.0, 982.75, 698.0, 549.0, -291.0, 397.0, -961.5
    .float 861.5, 384.0, 454.0, 892.25, -412.0, 653.75, 850.75, 607.5, 791.75, 78.25
    .float -438.5, 378.5, 823.5, 938.25, -637.25, 390.5, -857.25, 790.5, 988.0, 357.0
## DATA DEFINE END
size1: .word MatrixSize

.bss
.align 2
image_patch_buffer:
    .space 100    # 25 floats × 4 bytes
output_max:
     .space 12*12*8 #float

p:
    .space 40  #float

dense_output:
    .space  40  # 10 float values x 4 bytes filter

conv_output: 
     .space 8*24*24*4 # 8 channels, height and width 24x24 and word size 4 bytes


.section .data 
.align 2
# Fully Connected Layer Weights and Biases, All data based on a similar tensorflow CNN with 97% accuracy trained on unaugmented but normalized MNIST data 
.global weights
#  1152 x 10 Dense weights

## DENSE_WEIGHTS BEGIN
dense_weights:
    .float -0.019454, 0.052944, -0.019980, -0.059531, -0.027493, 0.188047, -0.095913, 0.018553, -0.058113, -0.001495  # dense_weights[0]
    .float -0.163107, 0.075439, 0.062300, 0.046593, -0.139581, 0.028529, -0.009515, 0.023709, -0.157005, 0.000629  # dense_weights[1]
    .float -0.053098, 0.050497, -0.037530, 0.110969, -0.098728, 0.045618, -0.036260, -0.059549, -0.136004, -0.012994  # dense_weights[2]
    .float -0.126564, 0.091490, 0.020178, -0.005992, -0.018001, 0.169720, 0.002427, -0.049300, -0.028794, 0.064321  # dense_weights[3]
    .float -0.125214, 0.175270, -0.085538, 0.158388, -0.103473, -0.099004, 0.166495, -0.062315, -0.093507, -0.036976  # dense_weights[4]
    .float -0.133388, 0.131646, -0.061291, 0.045177, 0.002343, -0.076320, 0.042856, -0.000385, -0.058441, 0.061463  # dense_weights[5]
    .float -0.049842, 0.054310, 0.036571, 0.062626, -0.137013, 0.055778, -0.018498, 0.039826, -0.096551, -0.053670  # dense_weights[6]
    .float -0.007044, 0.050315, -0.020077, 0.093836, -0.102154, -0.141186, -0.001240, -0.053397, -0.095064, -0.097373  # dense_weights[7]
    .float -0.136087, 0.008955, 0.040379, 0.046449, -0.042571, 0.035068, 0.091748, 0.038234, -0.113300, 0.025493  # dense_weights[8]
    .float -0.025987, -0.001953, -0.014813, 0.099743, -0.020420, -0.106643, 0.093919, -0.039231, -0.066247, -0.080750  # dense_weights[9]
    .float -0.027372, 0.001986, 0.018716, 0.125412, -0.019712, -0.049229, -0.031268, -0.098477, -0.105516, -0.105465  # dense_weights[10]
    .float -0.153193, 0.026512, 0.020419, 0.101017, -0.114620, 0.056939, 0.061976, -0.077923, -0.097978, 0.011068  # dense_weights[11]
    .float -0.058130, 0.019424, -0.060260, 0.126500, -0.051435, -0.181494, -0.007587, -0.156843, 0.010439, -0.213217  # dense_weights[12]
    .float -0.112732, -0.009108, -0.009286, -0.001113, -0.052361, -0.101286, 0.076132, -0.135361, 0.054953, 0.022090  # dense_weights[13]
    .float -0.072913, 0.002746, 0.000792, 0.029595, -0.095684, 0.006918, -0.016310, -0.006732, -0.069606, -0.046764  # dense_weights[14]
    .float -0.084544, 0.067474, -0.050773, 0.044391, -0.053472, -0.148974, 0.111544, -0.009691, 0.025407, -0.165307  # dense_weights[15]
    .float -0.077437, 0.174500, 0.001037, -0.042229, -0.034236, 0.032512, 0.002405, 0.041138, -0.105311, -0.074484  # dense_weights[16]
    .float -0.090269, -0.007810, 0.049543, 0.098680, -0.050116, -0.049642, 0.103190, -0.096873, 0.011065, -0.008149  # dense_weights[17]
    .float 0.002372, -0.006514, 0.068375, 0.067767, -0.017565, -0.049586, -0.019664, -0.115967, -0.008535, -0.146028  # dense_weights[18]
    .float -0.083967, -0.016499, 0.045036, 0.066564, -0.087506, -0.057222, 0.047834, -0.026835, -0.025131, -0.012406  # dense_weights[19]
    .float 0.051812, -0.037214, 0.031581, -0.034876, -0.076403, -0.100711, 0.093810, -0.053861, 0.085405, -0.144855  # dense_weights[20]
    .float -0.019810, -0.038552, 0.046501, 0.045358, -0.113008, -0.192753, 0.096594, -0.201128, -0.073358, -0.120560  # dense_weights[21]
    .float -0.046701, -0.073142, 0.097007, 0.048086, -0.200906, -0.073197, 0.133256, 0.022412, -0.032251, -0.109123  # dense_weights[22]
    .float -0.049786, -0.045999, -0.050210, 0.022576, -0.113171, -0.073744, 0.022107, 0.026343, 0.031574, -0.023247  # dense_weights[23]
    .float -0.107387, 0.061103, -0.044542, 0.043298, -0.079614, 0.004987, 0.152510, -0.019432, -0.084298, -0.081397  # dense_weights[24]
    .float 0.029141, -0.082725, 0.046490, 0.031198, -0.210564, -0.022780, 0.076276, -0.058894, -0.007528, 0.025734  # dense_weights[25]
    .float -0.057561, -0.077766, 0.029953, 0.022520, -0.106138, -0.006070, -0.042492, -0.077055, 0.040553, -0.028803  # dense_weights[26]
    .float -0.103999, -0.070248, 0.093722, 0.070224, -0.055845, -0.049567, 0.079777, -0.031004, 0.058604, -0.058509  # dense_weights[27]
    .float -0.056869, -0.110032, 0.044399, 0.022682, -0.044448, -0.119029, 0.079592, -0.052144, 0.034344, 0.000546  # dense_weights[28]
    .float -0.082705, -0.023273, 0.089427, 0.054338, -0.236549, -0.059220, 0.122483, -0.142343, -0.106290, -0.090583  # dense_weights[29]
    .float -0.060249, -0.019228, 0.044992, -0.009999, -0.194536, 0.048502, 0.140484, -0.117123, -0.102407, -0.047852  # dense_weights[30]
    .float 0.022929, -0.092238, 0.062998, 0.027696, -0.028732, -0.115532, 0.112846, -0.030635, 0.051868, -0.070409  # dense_weights[31]
    .float 0.001512, 0.153330, -0.009733, -0.040712, -0.092797, -0.019998, 0.119675, -0.109666, -0.070173, -0.014676  # dense_weights[32]
    .float -0.094389, -0.084472, 0.033470, 0.080143, -0.072533, -0.039883, 0.023342, -0.076267, 0.013246, 0.033101  # dense_weights[33]
    .float -0.061750, 0.008262, 0.108768, 0.041791, -0.036909, -0.069974, -0.083278, -0.056121, 0.097240, -0.005916  # dense_weights[34]
    .float -0.056901, -0.088462, -0.005139, -0.014288, -0.078482, 0.032212, 0.101382, -0.105692, -0.102204, 0.020642  # dense_weights[35]
    .float -0.076435, -0.034498, -0.026978, 0.097116, 0.000159, -0.117088, -0.093274, -0.037511, -0.032447, 0.067575  # dense_weights[36]
    .float -0.081167, 0.013896, 0.059615, 0.023910, -0.137488, -0.153294, 0.012404, -0.134600, -0.007078, -0.082886  # dense_weights[37]
    .float -0.072912, -0.003640, 0.093438, -0.004330, -0.075081, 0.054436, 0.105003, -0.136818, -0.037112, 0.024210  # dense_weights[38]
    .float -0.018029, -0.077672, 0.035044, 0.076199, -0.169839, 0.017052, 0.142453, 0.009110, 0.038834, -0.013748  # dense_weights[39]
    .float 0.030793, -0.010366, 0.060622, -0.005300, 0.010553, -0.073279, -0.002963, -0.117413, 0.005092, 0.005208  # dense_weights[40]
    .float 0.052380, 0.081078, -0.036327, 0.008128, -0.042082, -0.036374, -0.047074, -0.020972, 0.050703, -0.081496  # dense_weights[41]
    .float 0.051284, -0.017520, 0.024495, -0.032154, -0.124208, -0.050820, -0.051401, -0.100920, -0.045464, 0.040976  # dense_weights[42]
    .float -0.060537, 0.085683, 0.007521, -0.055897, -0.049975, -0.002670, 0.001058, -0.056487, -0.006237, 0.062828  # dense_weights[43]
    .float -0.018017, -0.001027, 0.064176, -0.042904, 0.066609, -0.044063, 0.035738, -0.058331, -0.082478, -0.075124  # dense_weights[44]
    .float -0.011255, 0.017372, 0.006081, 0.112197, -0.106300, -0.144712, 0.019642, -0.171702, -0.082919, -0.212454  # dense_weights[45]
    .float -0.093657, 0.131605, -0.018445, -0.025991, -0.065735, 0.002141, 0.187928, -0.064648, -0.106526, -0.126800  # dense_weights[46]
    .float 0.029216, 0.074014, 0.111534, 0.083324, -0.134743, -0.035653, 0.096908, -0.075305, 0.063960, 0.022595  # dense_weights[47]
    .float 0.081715, 0.041450, 0.052909, 0.011754, -0.009283, -0.115624, 0.013389, -0.023783, -0.022881, -0.134447  # dense_weights[48]
    .float 0.042964, 0.066428, 0.053227, 0.076323, -0.114658, 0.054545, 0.028402, -0.067414, 0.015393, -0.070715  # dense_weights[49]
    .float -0.055172, 0.114220, 0.083199, -0.009143, -0.027048, 0.012139, -0.071593, -0.075751, -0.052025, 0.048412  # dense_weights[50]
    .float -0.039881, 0.064717, 0.006239, -0.016630, 0.049632, 0.057305, 0.082176, -0.077819, 0.067294, -0.086267  # dense_weights[51]
    .float 0.052796, 0.025752, 0.088132, -0.024754, -0.014803, 0.070431, 0.054414, -0.013329, 0.070170, -0.089940  # dense_weights[52]
    .float -0.052502, -0.037601, 0.054050, 0.038866, -0.132785, -0.049341, 0.054992, -0.193680, -0.056490, -0.309018  # dense_weights[53]
    .float -0.060201, 0.133453, -0.110269, -0.012504, 0.006469, -0.058412, 0.142009, -0.103391, -0.075858, -0.240726  # dense_weights[54]
    .float 0.029875, 0.063230, -0.036206, 0.091748, 0.016861, -0.024024, 0.093811, 0.001510, 0.032461, -0.107109  # dense_weights[55]
    .float 0.025727, 0.039342, 0.036792, 0.082474, -0.064615, -0.059813, 0.059289, -0.021545, -0.038965, -0.069825  # dense_weights[56]
    .float 0.036659, -0.004805, -0.000940, 0.026615, -0.003403, 0.022564, 0.042523, -0.057207, 0.051224, 0.012691  # dense_weights[57]
    .float 0.017818, 0.045685, -0.055961, 0.092845, 0.006361, 0.063510, 0.042458, -0.051005, 0.010128, -0.103371  # dense_weights[58]
    .float 0.001035, 0.006601, 0.006450, -0.098689, -0.085623, 0.078018, 0.077414, -0.098693, 0.031466, -0.124141  # dense_weights[59]
    .float 0.078633, -0.002293, 0.089828, 0.062475, 0.016341, 0.045665, -0.039071, -0.105527, 0.064868, -0.041933  # dense_weights[60]
    .float -0.087038, -0.009776, -0.055263, 0.029485, -0.073875, -0.117743, 0.146928, -0.156855, 0.017735, -0.284489  # dense_weights[61]
    .float 0.000159, 0.051626, -0.185317, -0.075726, 0.038573, -0.024523, 0.225804, -0.189320, -0.023447, -0.270328  # dense_weights[62]
    .float 0.010947, 0.026946, -0.042038, -0.009531, -0.035824, 0.075031, 0.073343, -0.113266, -0.040366, -0.100663  # dense_weights[63]
    .float -0.070461, 0.027912, 0.005818, -0.005304, -0.042986, 0.087548, -0.020619, -0.164383, -0.028951, -0.066500  # dense_weights[64]
    .float -0.025728, -0.070156, -0.028421, -0.038778, -0.114430, 0.031514, 0.078115, -0.083569, -0.077286, -0.058811  # dense_weights[65]
    .float 0.014420, 0.042257, -0.044464, 0.029971, -0.110241, 0.033959, 0.052505, -0.137927, -0.045729, -0.078263  # dense_weights[66]
    .float -0.007460, -0.002733, -0.095498, -0.049729, -0.059116, 0.011528, 0.080232, -0.153943, 0.097984, -0.008241  # dense_weights[67]
    .float -0.020967, -0.077744, 0.058145, -0.006435, -0.032675, -0.016817, 0.029741, -0.213453, 0.002108, -0.083931  # dense_weights[68]
    .float -0.038535, -0.000780, -0.193986, -0.064259, 0.078388, 0.038528, 0.099026, -0.124457, 0.019976, -0.179152  # dense_weights[69]
    .float -0.033463, 0.078602, -0.264859, -0.209787, 0.097254, 0.093981, 0.165633, -0.226568, 0.030531, -0.188365  # dense_weights[70]
    .float 0.030160, -0.087710, 0.029794, 0.061809, -0.122600, 0.063178, 0.110320, -0.082733, -0.018136, -0.004473  # dense_weights[71]
    .float -0.082881, -0.042464, -0.030626, 0.051890, -0.109957, 0.006111, -0.016077, -0.171429, -0.042698, -0.095135  # dense_weights[72]
    .float -0.078900, -0.088396, -0.012248, -0.003013, -0.053957, 0.098891, 0.112283, -0.094671, -0.092985, -0.092987  # dense_weights[73]
    .float 0.067180, 0.023808, 0.070515, 0.035218, -0.073557, 0.018046, 0.015572, -0.214070, 0.029434, -0.120490  # dense_weights[74]
    .float 0.003838, -0.007858, -0.027892, -0.120585, 0.046726, -0.012981, 0.126571, -0.100669, 0.057141, -0.096604  # dense_weights[75]
    .float -0.038989, 0.009725, 0.068333, 0.012452, -0.038782, 0.017728, 0.036875, -0.213544, 0.052986, -0.061604  # dense_weights[76]
    .float -0.149735, -0.017332, -0.156810, -0.077128, 0.014799, -0.050851, 0.187220, -0.070137, -0.021990, -0.091081  # dense_weights[77]
    .float -0.062945, 0.115274, -0.162944, -0.172827, 0.044270, -0.020631, 0.137804, -0.052780, -0.020298, -0.060079  # dense_weights[78]
    .float 0.002971, -0.074847, -0.037061, -0.042535, -0.075248, 0.018844, 0.125184, -0.213092, 0.002538, -0.043168  # dense_weights[79]
    .float -0.043733, -0.031855, 0.021139, 0.003060, -0.067072, -0.011612, 0.035302, -0.087404, 0.027342, -0.068432  # dense_weights[80]
    .float 0.002975, -0.086547, -0.113216, -0.035760, -0.027975, 0.071829, 0.142131, -0.066075, -0.053472, -0.047622  # dense_weights[81]
    .float 0.033259, 0.025731, 0.019875, -0.064380, 0.052723, -0.000602, 0.050093, -0.125915, 0.002756, -0.107877  # dense_weights[82]
    .float -0.006064, 0.054875, 0.082332, -0.045644, 0.024492, 0.016569, 0.062075, -0.149738, -0.003430, -0.073718  # dense_weights[83]
    .float -0.056918, -0.030568, 0.017794, 0.020497, 0.082646, 0.082744, 0.125294, -0.152360, 0.053132, -0.122708  # dense_weights[84]
    .float -0.160900, -0.078962, -0.150204, -0.144652, -0.062136, 0.042073, 0.105245, -0.051772, -0.076261, -0.043995  # dense_weights[85]
    .float 0.022370, 0.013791, -0.040892, -0.039043, 0.025121, 0.122123, 0.073654, -0.025733, -0.040336, -0.012922  # dense_weights[86]
    .float -0.022470, 0.033507, -0.035606, -0.027528, 0.056206, -0.050037, 0.091281, -0.173332, 0.020733, -0.165828  # dense_weights[87]
    .float -0.062955, -0.026793, -0.024726, -0.003858, -0.037476, 0.097935, 0.164629, -0.128812, -0.105706, -0.053818  # dense_weights[88]
    .float -0.035841, -0.121690, -0.067689, -0.093389, -0.077013, 0.059266, 0.059942, -0.044620, 0.004491, -0.144186  # dense_weights[89]
    .float -0.031197, -0.085982, -0.123237, -0.031828, -0.024884, 0.022066, 0.016644, -0.099466, -0.084597, -0.149535  # dense_weights[90]
    .float -0.060716, -0.036009, -0.085370, -0.031096, 0.063001, 0.171307, 0.014233, -0.070559, -0.113721, -0.013559  # dense_weights[91]
    .float -0.039992, -0.064180, 0.016581, 0.046592, -0.048130, 0.053295, 0.043400, -0.143769, 0.008160, -0.118109  # dense_weights[92]
    .float -0.043524, 0.021302, -0.085163, -0.142242, -0.101986, 0.033888, 0.089395, -0.032922, 0.022701, 0.026554  # dense_weights[93]
    .float -0.105758, 0.062224, -0.020305, 0.022496, -0.027700, 0.113981, -0.017915, 0.067447, -0.091744, -0.043221  # dense_weights[94]
    .float -0.021558, 0.001111, -0.100569, -0.111441, 0.095251, 0.081649, 0.046969, -0.151803, 0.022982, -0.110065  # dense_weights[95]
    .float -0.023736, 0.056106, 0.085898, 0.029239, 0.000123, 0.183024, -0.089884, 0.035491, -0.114803, -0.019725  # dense_weights[96]
    .float -0.017185, 0.033221, 0.052694, 0.009335, 0.000982, -0.042362, -0.106398, -0.018914, -0.020156, -0.022524  # dense_weights[97]
    .float -0.018250, -0.018043, 0.053272, 0.038068, -0.019460, -0.139867, -0.058035, 0.006341, -0.046633, -0.026596  # dense_weights[98]
    .float -0.155619, 0.063549, -0.000396, 0.104935, -0.002319, 0.026332, 0.008569, -0.010355, -0.122698, 0.009645  # dense_weights[99]
    .float -0.034662, 0.122772, 0.034892, 0.131586, -0.006795, -0.224939, -0.061062, 0.020927, 0.070391, -0.210541  # dense_weights[100]
    .float 0.013362, 0.072545, -0.055487, 0.142447, -0.034469, -0.110391, 0.076623, 0.042395, -0.006148, -0.184982  # dense_weights[101]
    .float -0.117581, 0.075990, -0.056640, 0.075425, -0.087488, -0.037957, -0.037355, 0.014915, -0.152615, -0.105689  # dense_weights[102]
    .float -0.089429, 0.030976, 0.037877, 0.088266, 0.052317, -0.134317, -0.092016, 0.127821, -0.059004, -0.088698  # dense_weights[103]
    .float -0.045006, -0.035291, 0.093119, 0.041305, -0.099780, 0.014691, 0.026083, -0.036966, -0.189495, -0.080613  # dense_weights[104]
    .float -0.057135, 0.013303, 0.021212, 0.036280, 0.009916, 0.013225, -0.097956, -0.004673, -0.034228, -0.004226  # dense_weights[105]
    .float 0.043363, 0.018738, 0.022232, 0.053714, 0.011278, -0.120142, -0.017069, 0.037100, 0.052114, -0.020101  # dense_weights[106]
    .float 0.040154, 0.026590, -0.050279, 0.110760, -0.054683, 0.003056, -0.027921, 0.010320, -0.036402, -0.024927  # dense_weights[107]
    .float 0.064201, -0.007013, -0.016944, 0.080287, 0.017417, -0.116968, -0.117495, 0.016162, 0.016998, -0.067258  # dense_weights[108]
    .float 0.030945, 0.032762, -0.078464, 0.006092, 0.051668, -0.044726, 0.073584, 0.053840, -0.029480, -0.052941  # dense_weights[109]
    .float 0.008677, -0.045545, 0.007802, 0.061227, 0.036644, -0.019570, 0.038163, 0.017360, -0.017214, -0.027096  # dense_weights[110]
    .float 0.016832, 0.008694, -0.054642, -0.005652, -0.025711, -0.073519, 0.115135, 0.049044, 0.028779, -0.129239  # dense_weights[111]
    .float -0.167404, 0.067011, -0.164962, 0.037878, -0.017589, -0.038701, 0.103554, 0.051466, -0.128701, -0.143578  # dense_weights[112]
    .float -0.048499, -0.037097, -0.014116, 0.084297, -0.033299, -0.015841, -0.002160, -0.006551, -0.037754, 0.054114  # dense_weights[113]
    .float -0.012917, -0.064499, 0.042014, 0.044227, -0.045620, 0.016837, 0.020364, -0.010908, 0.045039, 0.007622  # dense_weights[114]
    .float -0.071887, -0.033979, -0.058441, -0.051788, 0.000788, 0.026009, -0.009070, 0.054378, -0.036420, -0.009035  # dense_weights[115]
    .float -0.135248, -0.019890, 0.041683, 0.008857, 0.113119, -0.002906, -0.123793, 0.060004, 0.060837, 0.071670  # dense_weights[116]
    .float -0.069983, -0.007103, 0.041959, 0.045204, 0.010481, -0.114021, -0.010133, 0.064424, -0.043432, -0.084357  # dense_weights[117]
    .float 0.027886, -0.117719, -0.031260, -0.057208, -0.016619, 0.066293, 0.032093, 0.062354, 0.021870, 0.036478  # dense_weights[118]
    .float -0.071885, 0.022671, 0.018674, -0.020985, 0.068621, -0.130310, 0.088078, 0.022305, -0.012151, -0.043396  # dense_weights[119]
    .float -0.108913, 0.117190, -0.146400, 0.075281, 0.211836, 0.039777, 0.055050, 0.041470, -0.129177, -0.252038  # dense_weights[120]
    .float 0.032753, -0.100677, 0.066144, 0.010053, -0.072031, 0.031582, -0.106482, -0.004035, -0.034276, -0.024279  # dense_weights[121]
    .float 0.018752, -0.099473, 0.000657, 0.091501, -0.095307, -0.019457, -0.078663, 0.073073, -0.067015, 0.081108  # dense_weights[122]
    .float -0.037666, -0.068292, 0.020958, 0.035429, -0.065396, -0.014430, 0.009954, -0.058456, 0.073492, 0.068956  # dense_weights[123]
    .float 0.052737, -0.093633, -0.062477, 0.091093, -0.034912, -0.127229, -0.106393, 0.082885, 0.034651, 0.018108  # dense_weights[124]
    .float -0.019425, 0.029650, 0.013674, 0.062907, 0.036809, -0.033396, -0.029716, 0.076533, -0.071037, -0.093383  # dense_weights[125]
    .float -0.073349, -0.127070, -0.061572, 0.056608, 0.020644, -0.010099, 0.084173, 0.051734, 0.039836, 0.011869  # dense_weights[126]
    .float -0.015169, -0.043164, 0.059619, 0.067769, -0.063751, -0.039657, -0.024477, 0.108606, 0.067830, 0.030363  # dense_weights[127]
    .float 0.008676, 0.204645, -0.159711, 0.003438, 0.233725, -0.054297, 0.036058, 0.011786, -0.172541, -0.274032  # dense_weights[128]
    .float 0.009972, 0.006470, -0.027857, -0.019870, -0.089424, 0.055009, -0.028204, -0.031583, 0.023617, -0.010024  # dense_weights[129]
    .float 0.012502, -0.057352, -0.075580, 0.041417, -0.000730, 0.059388, -0.077427, 0.086304, 0.011417, -0.032527  # dense_weights[130]
    .float -0.046638, -0.026742, 0.016282, 0.008462, -0.001110, -0.026820, 0.016201, -0.069575, 0.056042, 0.053463  # dense_weights[131]
    .float 0.016181, -0.164212, 0.041141, 0.096134, 0.008679, -0.028716, -0.101539, 0.074949, -0.035884, -0.027433  # dense_weights[132]
    .float -0.098587, 0.003299, 0.021696, 0.089082, 0.012629, -0.023519, -0.089711, 0.047876, 0.051001, -0.234372  # dense_weights[133]
    .float 0.023752, 0.059204, -0.136539, -0.073923, 0.020321, 0.107815, 0.150007, -0.005679, 0.049946, -0.032780  # dense_weights[134]
    .float -0.079556, -0.108772, 0.035838, 0.019442, 0.012757, -0.004852, 0.059616, -0.018093, -0.099600, 0.062132  # dense_weights[135]
    .float 0.030750, 0.070181, -0.021342, 0.025800, 0.176311, -0.119955, -0.036346, -0.002847, -0.139379, -0.046335  # dense_weights[136]
    .float 0.006545, 0.030830, 0.022122, 0.118678, -0.019350, 0.078490, -0.016868, 0.021073, 0.026240, -0.032995  # dense_weights[137]
    .float -0.035114, -0.137557, 0.016098, 0.104520, -0.048038, 0.042209, -0.118376, 0.083898, -0.003074, 0.101680  # dense_weights[138]
    .float 0.001425, -0.018320, -0.107638, 0.023547, -0.071222, 0.056422, 0.014049, -0.011458, 0.089842, -0.054104  # dense_weights[139]
    .float -0.097118, -0.138324, -0.000428, 0.038947, -0.172560, -0.015142, -0.077102, -0.016479, -0.013434, 0.071830  # dense_weights[140]
    .float -0.059689, -0.046909, 0.039095, 0.149826, 0.009366, -0.138522, -0.138365, -0.108733, 0.020094, -0.165934  # dense_weights[141]
    .float 0.007721, 0.118772, -0.144671, -0.089499, 0.135990, -0.062809, 0.129307, -0.043854, 0.077179, -0.082456  # dense_weights[142]
    .float -0.007445, 0.009415, -0.031260, 0.025822, -0.098834, -0.025472, -0.000170, -0.014371, 0.019776, -0.044531  # dense_weights[143]
    .float 0.095407, 0.079804, -0.069748, 0.041856, 0.040284, -0.176392, -0.045393, -0.123225, -0.082889, 0.047276  # dense_weights[144]
    .float -0.020001, -0.066126, -0.047870, 0.081074, -0.012927, 0.039454, -0.061688, 0.078775, 0.082465, 0.059075  # dense_weights[145]
    .float -0.045605, -0.124761, -0.044712, 0.060761, -0.044232, 0.059890, -0.132264, 0.059762, 0.024685, 0.005173  # dense_weights[146]
    .float -0.036296, 0.016711, -0.103089, -0.092568, -0.105545, 0.025854, 0.090222, -0.121748, 0.022856, -0.054951  # dense_weights[147]
    .float -0.074245, -0.100124, 0.030715, 0.036176, -0.077420, 0.062097, -0.029321, 0.016161, 0.002874, -0.001685  # dense_weights[148]
    .float -0.050673, -0.139876, 0.124524, 0.142020, 0.007756, -0.027270, -0.102880, -0.107108, 0.016287, -0.019220  # dense_weights[149]
    .float 0.035856, 0.098406, -0.190850, -0.115640, 0.105096, -0.028706, 0.172135, -0.252318, -0.002035, -0.119596  # dense_weights[150]
    .float 0.003817, 0.011911, -0.062785, 0.030051, -0.055387, -0.036012, -0.026022, -0.021964, -0.013966, 0.057846  # dense_weights[151]
    .float 0.024943, 0.125746, 0.018685, 0.008777, 0.001340, -0.145511, -0.075637, 0.003048, -0.014615, 0.046146  # dense_weights[152]
    .float 0.024017, -0.010129, -0.049003, 0.042579, -0.040403, 0.113789, -0.084204, 0.037728, 0.097657, 0.015954  # dense_weights[153]
    .float 0.046613, -0.072648, 0.017953, -0.045481, -0.096062, 0.041056, -0.002933, 0.003488, 0.063229, 0.015174  # dense_weights[154]
    .float -0.018546, -0.025989, -0.064804, -0.127643, -0.026119, 0.032495, 0.086716, -0.031903, 0.036683, 0.042313  # dense_weights[155]
    .float 0.027697, -0.064549, 0.035737, 0.027802, -0.064571, -0.018607, -0.013212, 0.013780, -0.031066, 0.094220  # dense_weights[156]
    .float -0.041875, -0.156833, 0.053457, 0.013393, -0.041451, -0.046793, -0.037141, -0.038916, 0.022874, 0.011317  # dense_weights[157]
    .float -0.091511, 0.207348, -0.285654, -0.263545, 0.344415, -0.067827, 0.226594, -0.187490, 0.024566, -0.146726  # dense_weights[158]
    .float 0.015757, 0.021862, -0.001008, 0.084803, -0.090674, 0.058852, 0.010394, -0.083545, -0.006235, 0.035015  # dense_weights[159]
    .float 0.023801, 0.034183, -0.018921, -0.001158, 0.007331, -0.064893, -0.118979, 0.000987, 0.040067, 0.011421  # dense_weights[160]
    .float 0.019875, -0.079114, 0.017357, 0.031330, -0.083166, 0.136190, -0.120068, -0.050454, 0.003316, -0.017222  # dense_weights[161]
    .float -0.022778, -0.061180, 0.043921, -0.069372, -0.096567, 0.059679, -0.038855, -0.023209, -0.024451, -0.015628  # dense_weights[162]
    .float -0.066829, -0.020888, -0.113772, -0.145439, 0.072645, 0.134591, 0.028735, 0.048382, -0.009103, 0.063753  # dense_weights[163]
    .float 0.022110, -0.038772, 0.011999, -0.015567, -0.131235, 0.097326, 0.055530, -0.014828, 0.078968, -0.008856  # dense_weights[164]
    .float -0.057533, -0.177519, 0.035716, -0.071082, -0.007908, 0.061735, -0.015007, -0.152092, 0.046591, 0.000691  # dense_weights[165]
    .float -0.096536, 0.281279, -0.411483, -0.287590, 0.262520, -0.048950, 0.172098, -0.216838, -0.086079, -0.098228  # dense_weights[166]
    .float -0.031578, 0.035236, -0.021569, 0.090987, -0.026007, 0.054071, -0.029854, 0.067399, -0.010846, 0.002340  # dense_weights[167]
    .float -0.036071, 0.058392, 0.005838, 0.078052, -0.031143, 0.070279, -0.062205, -0.057336, -0.001467, -0.022767  # dense_weights[168]
    .float 0.060407, -0.047103, 0.059943, -0.009131, -0.063697, 0.082344, 0.000736, -0.003223, 0.048332, -0.056004  # dense_weights[169]
    .float 0.068712, 0.052099, -0.038737, -0.063863, -0.022822, -0.012748, -0.078125, 0.048613, 0.072245, -0.003160  # dense_weights[170]
    .float -0.103480, 0.003670, -0.226509, -0.215224, 0.035349, 0.180355, -0.045677, 0.016395, 0.044907, -0.101442  # dense_weights[171]
    .float 0.100525, 0.004621, -0.039420, -0.000950, -0.013798, -0.000159, 0.022440, 0.003398, 0.044963, -0.003619  # dense_weights[172]
    .float -0.048246, -0.118354, -0.051534, -0.066472, 0.045318, 0.115917, 0.066151, -0.121165, 0.056645, -0.082868  # dense_weights[173]
    .float -0.006972, 0.215512, -0.280180, -0.185308, 0.181018, 0.046845, -0.052539, -0.051485, -0.076972, -0.108344  # dense_weights[174]
    .float 0.010850, -0.046801, -0.041327, -0.029303, -0.076578, 0.047709, 0.044969, -0.063583, 0.024181, 0.027950  # dense_weights[175]
    .float 0.080678, 0.054519, 0.123213, -0.044672, -0.041479, -0.078466, -0.079365, -0.012495, -0.024240, -0.026958  # dense_weights[176]
    .float 0.075637, 0.008303, 0.062097, -0.088270, -0.029747, -0.055407, -0.040286, -0.100323, 0.070095, -0.081135  # dense_weights[177]
    .float 0.005574, -0.003418, -0.045079, -0.001881, -0.024370, 0.002801, -0.072420, -0.089728, -0.032175, -0.053488  # dense_weights[178]
    .float -0.112257, 0.066707, -0.132958, -0.077474, 0.014558, 0.117871, 0.029312, -0.032723, -0.037523, -0.071687  # dense_weights[179]
    .float 0.056761, -0.071299, -0.021827, 0.024532, 0.070057, 0.059255, 0.066495, -0.096513, 0.060072, -0.057382  # dense_weights[180]
    .float -0.086340, -0.169191, -0.170621, -0.183736, 0.104371, 0.152889, 0.113330, -0.210907, 0.029246, -0.285766  # dense_weights[181]
    .float -0.181400, 0.094073, -0.131970, -0.147849, 0.246798, 0.133309, -0.041912, 0.015606, -0.100690, -0.166921  # dense_weights[182]
    .float -0.006550, -0.034192, 0.088739, 0.001301, -0.056665, -0.040956, 0.073509, -0.082634, 0.039647, -0.043994  # dense_weights[183]
    .float 0.119026, -0.058442, -0.074885, -0.036568, -0.035869, -0.000051, 0.078020, -0.140178, -0.061585, -0.013725  # dense_weights[184]
    .float -0.026977, -0.118987, -0.004898, 0.065680, -0.028729, 0.001286, -0.018270, -0.196245, -0.040675, -0.055505  # dense_weights[185]
    .float 0.099989, -0.099838, -0.017114, -0.054479, 0.030279, 0.026690, 0.034026, -0.167482, 0.055278, -0.044688  # dense_weights[186]
    .float -0.234934, -0.085150, -0.117597, -0.148662, -0.031557, 0.204706, -0.115439, -0.048043, 0.074780, -0.094190  # dense_weights[187]
    .float -0.041098, -0.019277, 0.025559, -0.093464, -0.003137, 0.077086, 0.141049, -0.022153, -0.029229, -0.048724  # dense_weights[188]
    .float -0.112016, -0.037408, -0.215637, -0.171725, 0.104368, 0.249510, 0.057376, -0.181091, 0.041435, -0.208097  # dense_weights[189]
    .float -0.205486, -0.070093, 0.080184, -0.089640, 0.001483, 0.189331, -0.108490, 0.055264, -0.010235, -0.024511  # dense_weights[190]
    .float -0.007992, -0.004833, -0.049210, -0.017789, 0.014039, -0.030828, 0.072112, -0.061792, 0.071628, -0.114498  # dense_weights[191]
    .float -0.122013, -0.004756, 0.024665, -0.010215, -0.086003, 0.168381, -0.111702, 0.047570, -0.101555, -0.091131  # dense_weights[192]
    .float -0.103230, -0.104103, 0.031081, 0.021398, -0.056162, -0.070671, 0.007228, 0.089680, -0.061106, -0.058364  # dense_weights[193]
    .float -0.003467, -0.109153, 0.002662, 0.061949, -0.117494, -0.210681, -0.087345, -0.004540, -0.060690, -0.059637  # dense_weights[194]
    .float -0.010649, 0.033678, -0.039881, 0.051049, -0.019454, -0.135883, 0.011914, 0.004349, -0.001381, -0.137735  # dense_weights[195]
    .float -0.047598, -0.038445, -0.096788, 0.115433, 0.023047, -0.256048, -0.017625, 0.005391, 0.096527, -0.118866  # dense_weights[196]
    .float -0.052429, 0.052589, 0.001623, -0.000940, -0.017008, -0.109175, 0.005170, 0.088905, 0.035976, -0.044058  # dense_weights[197]
    .float 0.002827, 0.006098, -0.010224, 0.063484, 0.019486, -0.048838, 0.000209, 0.007971, -0.048124, -0.017199  # dense_weights[198]
    .float -0.081395, 0.008071, -0.016797, 0.014481, 0.047185, -0.203838, -0.012859, 0.016139, 0.001527, -0.136964  # dense_weights[199]
    .float -0.170486, -0.011666, 0.052234, -0.007564, -0.025027, 0.060990, 0.009919, 0.145310, -0.213568, -0.224179  # dense_weights[200]
    .float -0.082670, -0.020979, 0.064096, -0.013021, -0.004997, -0.037259, -0.010257, -0.009875, 0.019255, 0.020874  # dense_weights[201]
    .float -0.016190, -0.155044, 0.081616, 0.034866, -0.115853, -0.098001, -0.100093, -0.002104, 0.069089, 0.045462  # dense_weights[202]
    .float 0.034115, -0.024817, 0.049391, -0.027945, -0.033508, -0.115527, -0.030817, 0.044088, 0.009536, 0.005094  # dense_weights[203]
    .float -0.003351, 0.013562, 0.045804, 0.178893, -0.114245, -0.135916, -0.199201, 0.016464, 0.024154, -0.006036  # dense_weights[204]
    .float -0.047454, -0.008061, 0.037504, -0.018263, 0.053795, -0.097929, -0.033752, -0.010971, -0.031723, 0.006866  # dense_weights[205]
    .float -0.080076, -0.030094, 0.033236, 0.044173, -0.101366, -0.011099, -0.056266, -0.029029, 0.038819, 0.016417  # dense_weights[206]
    .float -0.018018, 0.019250, 0.034181, 0.065375, 0.003162, -0.109392, -0.019816, -0.024664, -0.044772, -0.019525  # dense_weights[207]
    .float -0.135839, 0.056151, -0.013001, -0.058110, 0.104703, 0.002552, 0.108816, 0.241182, -0.236770, -0.424230  # dense_weights[208]
    .float -0.039785, -0.061188, -0.049822, -0.016329, -0.032535, -0.044849, -0.003563, -0.012161, -0.070822, 0.047518  # dense_weights[209]
    .float 0.024003, -0.014222, -0.015578, 0.111674, -0.015058, -0.066325, -0.106611, 0.077698, -0.079038, -0.023612  # dense_weights[210]
    .float 0.022990, -0.024647, 0.014315, 0.035770, -0.119623, 0.012058, -0.042362, -0.005183, -0.024789, -0.001094  # dense_weights[211]
    .float -0.031540, -0.024383, -0.000743, 0.213109, -0.110972, -0.122934, -0.091220, 0.092147, -0.079045, -0.007470  # dense_weights[212]
    .float -0.077595, -0.039845, -0.014554, -0.022872, 0.038172, -0.081047, -0.036498, 0.040678, -0.028010, -0.129195  # dense_weights[213]
    .float -0.047186, -0.023883, -0.028664, 0.021039, -0.055118, 0.012998, 0.053704, 0.009017, 0.018880, -0.000825  # dense_weights[214]
    .float -0.063237, -0.028699, -0.022650, 0.001936, -0.060496, -0.108552, 0.002532, 0.101756, 0.009191, -0.027469  # dense_weights[215]
    .float -0.064900, 0.092814, -0.236394, -0.174845, 0.169760, 0.163296, 0.009225, 0.250895, -0.130572, -0.490092  # dense_weights[216]
    .float 0.014501, -0.038878, -0.008726, -0.014356, -0.062844, -0.037844, 0.014831, 0.121374, 0.028272, 0.005101  # dense_weights[217]
    .float 0.019369, -0.083516, 0.024839, 0.071488, -0.044889, 0.011176, -0.033586, 0.016524, -0.091617, 0.016853  # dense_weights[218]
    .float 0.066578, 0.004767, 0.039333, 0.038017, -0.135577, -0.025966, 0.041253, -0.039941, 0.043631, 0.068974  # dense_weights[219]
    .float -0.014454, -0.046665, 0.043453, 0.190563, -0.028694, -0.086138, -0.070834, 0.011211, 0.081647, 0.026568  # dense_weights[220]
    .float -0.091625, 0.011478, -0.000221, 0.077303, 0.044658, -0.108712, -0.082257, 0.057688, 0.013438, -0.079794  # dense_weights[221]
    .float -0.000696, -0.060668, 0.066397, -0.034669, 0.059522, 0.062994, 0.052000, -0.022070, 0.034531, 0.001690  # dense_weights[222]
    .float 0.046642, -0.055680, -0.053547, 0.062058, -0.019312, -0.036036, -0.072690, -0.011352, -0.024928, 0.032446  # dense_weights[223]
    .float -0.088354, 0.123311, -0.098640, -0.095516, 0.346760, 0.109027, 0.099087, 0.127489, -0.054776, -0.550025  # dense_weights[224]
    .float -0.047162, -0.069512, 0.044791, -0.000700, -0.139233, 0.071678, -0.117626, 0.019511, 0.015757, 0.032261  # dense_weights[225]
    .float 0.002316, -0.090659, -0.009903, 0.103681, -0.139310, 0.016730, -0.114693, 0.044908, -0.063003, 0.061192  # dense_weights[226]
    .float -0.024641, -0.034160, -0.084981, 0.053215, -0.055159, -0.077260, 0.090816, -0.040330, 0.046109, 0.060534  # dense_weights[227]
    .float -0.085791, -0.136354, 0.069240, 0.093165, -0.037793, -0.081096, -0.118912, -0.007948, 0.071873, -0.026101  # dense_weights[228]
    .float -0.054869, -0.104179, 0.032616, 0.082584, -0.032641, -0.084550, -0.084776, 0.045249, 0.016936, -0.049776  # dense_weights[229]
    .float -0.014043, 0.001508, -0.067796, -0.113976, 0.101223, 0.031658, 0.163366, -0.088504, 0.105311, -0.001297  # dense_weights[230]
    .float 0.028768, -0.046883, -0.031847, -0.068058, -0.029425, 0.061190, 0.008259, -0.020144, 0.076483, -0.085500  # dense_weights[231]
    .float -0.017533, 0.138023, 0.033376, -0.007552, 0.290175, -0.088737, -0.030272, 0.074643, -0.033238, -0.288031  # dense_weights[232]
    .float 0.047981, 0.044001, 0.053259, -0.004539, -0.205414, -0.022169, -0.108270, 0.053176, -0.066549, 0.037708  # dense_weights[233]
    .float -0.075367, -0.045330, 0.088345, 0.076947, -0.084214, -0.002871, -0.004016, 0.097735, 0.025495, 0.046454  # dense_weights[234]
    .float -0.013706, -0.126425, -0.018524, 0.045911, -0.028886, -0.061646, 0.103943, -0.031984, -0.010906, 0.134879  # dense_weights[235]
    .float -0.028277, -0.072821, 0.036830, 0.028771, -0.089927, -0.031988, -0.161971, 0.050339, 0.031735, 0.015962  # dense_weights[236]
    .float -0.042400, -0.056435, 0.076367, 0.050327, -0.159019, -0.078580, -0.228118, 0.047379, 0.028476, 0.044969  # dense_weights[237]
    .float -0.075847, 0.042778, -0.084009, -0.166953, 0.174840, 0.083175, 0.132240, -0.197939, -0.037323, -0.019415  # dense_weights[238]
    .float -0.021842, 0.056239, 0.069070, -0.001240, -0.005672, 0.073953, 0.003654, 0.016937, -0.012765, 0.043095  # dense_weights[239]
    .float 0.069858, 0.157012, 0.079207, 0.102262, 0.187741, -0.315889, -0.149444, -0.039166, -0.076889, -0.081069  # dense_weights[240]
    .float 0.105185, 0.019872, 0.040988, 0.045880, -0.175722, -0.023833, -0.152322, 0.102872, -0.070441, 0.009178  # dense_weights[241]
    .float 0.031785, 0.003831, 0.075290, 0.044766, -0.135272, -0.012995, -0.103714, 0.094630, -0.035736, -0.017585  # dense_weights[242]
    .float -0.104831, -0.054534, 0.010608, 0.165065, -0.010202, -0.127812, 0.043911, -0.057840, -0.007452, 0.051272  # dense_weights[243]
    .float 0.031462, -0.108229, -0.017092, -0.026112, -0.093521, -0.095205, -0.128042, 0.070438, 0.112909, 0.012607  # dense_weights[244]
    .float -0.008553, -0.108878, 0.143990, 0.173216, -0.213345, -0.080867, -0.200601, -0.046470, 0.070061, 0.095857  # dense_weights[245]
    .float 0.016502, 0.124763, -0.128405, -0.163152, 0.350611, -0.054458, 0.203596, -0.267426, -0.046161, -0.085445  # dense_weights[246]
    .float 0.058330, 0.120862, -0.021280, -0.065104, -0.077252, 0.035435, 0.001990, 0.003418, 0.014419, 0.027360  # dense_weights[247]
    .float 0.111970, 0.039346, 0.070497, 0.048805, 0.178573, -0.215356, -0.257953, -0.075123, 0.024090, -0.040530  # dense_weights[248]
    .float -0.014987, -0.119303, 0.017259, 0.001061, -0.089277, 0.098320, -0.100286, 0.104466, 0.007444, 0.113373  # dense_weights[249]
    .float 0.068698, -0.086084, -0.031917, -0.025444, -0.193904, 0.132051, -0.031373, 0.039090, -0.011399, 0.105292  # dense_weights[250]
    .float -0.106670, -0.009735, -0.144970, 0.034265, 0.101296, -0.123780, -0.025720, -0.009589, 0.043587, 0.009231  # dense_weights[251]
    .float 0.045010, -0.008388, 0.042456, -0.032964, -0.137765, -0.018447, -0.127901, -0.032086, -0.003741, -0.063499  # dense_weights[252]
    .float 0.024747, -0.205106, 0.115744, 0.058166, -0.246144, 0.110214, -0.056481, -0.037174, 0.010203, 0.112390  # dense_weights[253]
    .float -0.027414, 0.229366, -0.294381, -0.142212, 0.386533, -0.048537, 0.037342, -0.150139, -0.192075, -0.264928  # dense_weights[254]
    .float -0.040654, -0.016803, -0.024055, 0.068320, 0.012225, -0.099027, -0.007614, -0.002849, -0.071947, 0.048451  # dense_weights[255]
    .float 0.131565, 0.003480, 0.095021, 0.031384, 0.054928, -0.088499, -0.300637, 0.047396, 0.091044, 0.027356  # dense_weights[256]
    .float -0.000286, -0.079341, -0.047664, 0.034178, -0.047770, 0.028434, -0.112611, 0.047769, 0.005786, -0.026976  # dense_weights[257]
    .float 0.018109, -0.094319, -0.179820, -0.076842, -0.066365, 0.233986, 0.038776, 0.016456, 0.025917, -0.014200  # dense_weights[258]
    .float -0.111074, 0.063618, -0.142278, -0.041228, 0.062978, -0.007054, -0.043871, 0.108688, 0.044883, -0.087971  # dense_weights[259]
    .float 0.055804, -0.108142, 0.051147, -0.086063, -0.073760, 0.028192, -0.072184, 0.001922, 0.027568, -0.024474  # dense_weights[260]
    .float 0.003701, -0.296115, 0.095207, 0.008197, -0.166288, 0.121246, -0.003051, -0.049456, 0.057971, 0.049087  # dense_weights[261]
    .float -0.212465, 0.362834, -0.429814, -0.094119, 0.412635, -0.083200, -0.033245, -0.062285, -0.059626, -0.364109  # dense_weights[262]
    .float -0.001792, 0.029560, 0.023753, -0.039724, -0.032410, 0.023856, -0.153587, -0.001727, 0.049056, 0.048211  # dense_weights[263]
    .float 0.060735, 0.033946, 0.068913, 0.021356, -0.084562, -0.120133, -0.183921, 0.070418, 0.084797, -0.002922  # dense_weights[264]
    .float 0.008206, -0.064331, -0.087885, -0.005298, -0.019859, 0.087132, 0.014209, 0.045038, 0.022460, -0.037678  # dense_weights[265]
    .float -0.048701, -0.082371, -0.160586, -0.083985, -0.104188, 0.184263, 0.009279, -0.007181, -0.017217, 0.052337  # dense_weights[266]
    .float -0.197388, 0.033637, -0.085711, -0.096009, 0.155987, 0.159217, -0.046445, -0.007949, -0.061829, -0.243938  # dense_weights[267]
    .float 0.044279, -0.060463, -0.011613, -0.076602, 0.021427, 0.080041, -0.000456, -0.047078, 0.005891, -0.021224  # dense_weights[268]
    .float 0.045229, -0.203501, -0.023340, -0.128457, -0.197750, 0.126352, 0.014076, -0.071462, 0.031508, 0.079540  # dense_weights[269]
    .float -0.069603, 0.195485, -0.265554, -0.159891, 0.296743, 0.037636, -0.081103, 0.079381, -0.239695, -0.390898  # dense_weights[270]
    .float 0.087931, -0.038223, -0.019228, 0.070890, -0.050599, -0.049063, -0.056547, 0.019603, 0.057967, 0.055095  # dense_weights[271]
    .float 0.118300, -0.054038, 0.094070, 0.017138, -0.116973, -0.153518, -0.101868, -0.065058, 0.019051, 0.020707  # dense_weights[272]
    .float 0.044681, -0.118841, -0.109212, -0.085157, -0.014049, 0.119505, -0.044546, -0.019661, -0.013776, 0.017429  # dense_weights[273]
    .float -0.050066, -0.109055, -0.095067, -0.125040, -0.073068, 0.145189, -0.031592, -0.004783, 0.074204, 0.010341  # dense_weights[274]
    .float -0.184203, -0.092370, -0.209640, -0.221051, 0.195778, 0.265205, -0.098726, 0.019460, -0.015722, -0.270856  # dense_weights[275]
    .float 0.022492, -0.043589, -0.028040, 0.009190, -0.006577, 0.048917, -0.053436, -0.112204, 0.044039, -0.016589  # dense_weights[276]
    .float -0.052216, -0.106342, -0.180862, -0.253655, 0.055552, 0.092100, 0.136256, -0.130949, 0.131514, -0.024495  # dense_weights[277]
    .float -0.238676, 0.027653, -0.059434, 0.024809, 0.183899, 0.165176, -0.241034, 0.112617, -0.000315, -0.299322  # dense_weights[278]
    .float 0.077007, -0.025658, 0.016026, -0.023752, 0.035955, -0.086154, -0.013513, -0.019351, 0.017986, -0.027412  # dense_weights[279]
    .float 0.063122, -0.025773, 0.000463, 0.087938, -0.029790, 0.023720, -0.000696, -0.020057, 0.043820, 0.003016  # dense_weights[280]
    .float 0.020832, -0.044294, -0.025102, -0.110701, -0.038738, 0.039866, -0.015507, -0.110957, -0.041866, -0.074319  # dense_weights[281]
    .float -0.039525, -0.093348, -0.094590, -0.112903, -0.086180, 0.145106, -0.122713, -0.105258, 0.010135, -0.000092  # dense_weights[282]
    .float -0.318891, -0.128402, -0.130035, -0.206523, 0.064517, 0.369281, -0.176619, -0.087420, -0.012734, -0.117159  # dense_weights[283]
    .float -0.002226, -0.123960, -0.009529, -0.090547, 0.079178, 0.055995, 0.019329, -0.042873, -0.000276, -0.097023  # dense_weights[284]
    .float -0.080382, -0.082720, -0.296776, -0.104646, 0.082761, 0.139428, 0.106323, -0.170903, -0.041525, -0.127512  # dense_weights[285]
    .float -0.131351, 0.070786, 0.035776, -0.010734, -0.050349, 0.196230, -0.212116, 0.044739, -0.017740, -0.078656  # dense_weights[286]
    .float -0.012262, -0.025369, -0.091158, -0.015251, -0.042283, 0.057469, 0.024357, -0.117102, -0.016472, -0.009209  # dense_weights[287]
    .float -0.067327, -0.022791, 0.051361, -0.025152, -0.059121, 0.189857, -0.010312, 0.171412, -0.147604, -0.083759  # dense_weights[288]
    .float -0.079078, -0.090495, -0.001114, 0.001692, -0.106070, -0.089757, -0.061197, 0.047220, -0.070953, 0.022414  # dense_weights[289]
    .float -0.004795, -0.127304, 0.030026, 0.142766, -0.142378, -0.182286, -0.118346, 0.018565, -0.080978, -0.106740  # dense_weights[290]
    .float -0.128839, 0.018492, 0.029042, -0.001911, -0.034436, -0.038209, -0.072586, 0.065711, 0.003830, -0.028254  # dense_weights[291]
    .float -0.109882, -0.140159, 0.090862, 0.102027, -0.166687, -0.141897, -0.224888, 0.073988, 0.055871, -0.037110  # dense_weights[292]
    .float -0.015575, -0.046788, 0.040476, 0.101383, 0.022195, -0.164349, -0.103857, -0.026806, -0.022771, -0.051558  # dense_weights[293]
    .float -0.060513, -0.087471, -0.005882, 0.073829, -0.105254, 0.027204, -0.026602, -0.034942, -0.046673, 0.006094  # dense_weights[294]
    .float -0.081714, -0.107154, 0.062679, 0.041327, -0.082881, -0.176754, -0.133163, 0.087905, 0.017378, -0.116834  # dense_weights[295]
    .float -0.140823, -0.008846, 0.035299, 0.001656, -0.092100, 0.097959, 0.080996, 0.097722, -0.159567, -0.168572  # dense_weights[296]
    .float 0.057317, -0.001516, 0.036251, 0.023795, -0.106647, -0.057127, 0.000209, -0.051982, -0.064118, 0.099603  # dense_weights[297]
    .float -0.039226, -0.147819, 0.012023, 0.062907, -0.084609, -0.037923, -0.072637, 0.028189, -0.023461, -0.004258  # dense_weights[298]
    .float 0.010187, -0.023541, 0.024143, -0.021157, -0.051097, -0.011636, -0.018016, -0.081956, 0.070390, 0.061257  # dense_weights[299]
    .float -0.047164, -0.071721, 0.169627, 0.126515, -0.153154, -0.246410, -0.175213, 0.005360, -0.032078, -0.042952  # dense_weights[300]
    .float -0.013743, -0.098740, 0.019190, 0.031343, -0.098092, -0.074531, -0.058039, 0.087358, -0.012628, -0.037953  # dense_weights[301]
    .float -0.017892, -0.107379, 0.005432, 0.030854, -0.134916, -0.019955, -0.007129, -0.047469, 0.031305, 0.028762  # dense_weights[302]
    .float -0.096071, -0.024820, 0.047342, 0.012708, 0.061779, -0.017342, 0.003336, -0.003865, -0.038836, -0.023999  # dense_weights[303]
    .float -0.147173, 0.103067, -0.135540, -0.314679, 0.183249, 0.084577, 0.014663, 0.214500, -0.048167, -0.328066  # dense_weights[304]
    .float 0.002035, -0.029179, -0.058309, 0.019585, 0.007762, -0.089630, 0.010473, -0.019249, -0.076705, 0.057776  # dense_weights[305]
    .float -0.079152, 0.002980, 0.033967, 0.149130, -0.064292, -0.054308, -0.050745, 0.079330, 0.017642, -0.033451  # dense_weights[306]
    .float 0.003248, 0.031013, -0.051783, 0.069489, -0.069908, -0.040148, -0.039507, -0.088787, 0.051912, -0.002089  # dense_weights[307]
    .float -0.143640, -0.055413, 0.072147, 0.181353, -0.053402, -0.061426, -0.125981, -0.022479, 0.045971, -0.127223  # dense_weights[308]
    .float -0.055442, -0.095224, -0.034699, 0.020541, -0.012140, -0.012073, -0.001565, 0.025370, 0.081475, 0.018741  # dense_weights[309]
    .float -0.043455, -0.094252, -0.029148, -0.075277, -0.002861, 0.039582, 0.077805, -0.059161, 0.037989, 0.102494  # dense_weights[310]
    .float -0.021645, -0.137390, -0.060078, -0.053464, 0.039323, 0.020122, -0.072600, -0.048928, 0.031450, 0.021897  # dense_weights[311]
    .float -0.111963, -0.019709, -0.205518, -0.455474, 0.278540, 0.150376, 0.159052, 0.224372, 0.155236, -0.220077  # dense_weights[312]
    .float 0.021813, 0.057792, -0.031698, 0.044433, -0.015719, 0.046530, -0.032094, 0.062238, -0.065903, -0.054823  # dense_weights[313]
    .float 0.036644, 0.034112, 0.088802, 0.061777, 0.032097, -0.075776, -0.092866, 0.111333, -0.040954, 0.006688  # dense_weights[314]
    .float -0.037534, 0.044738, 0.007764, 0.061892, 0.002060, -0.032181, 0.074788, -0.033129, -0.005818, 0.101745  # dense_weights[315]
    .float -0.115895, -0.107318, 0.020802, 0.161605, -0.009757, -0.073854, -0.075654, 0.024995, 0.162598, 0.113772  # dense_weights[316]
    .float -0.091291, 0.062472, -0.012817, 0.038121, -0.095349, 0.017831, -0.068333, 0.140290, -0.024746, 0.015903  # dense_weights[317]
    .float -0.017718, -0.029273, -0.041499, -0.063916, 0.062381, 0.044234, 0.117062, -0.033458, 0.036971, 0.077486  # dense_weights[318]
    .float 0.044483, 0.005400, -0.103339, -0.057935, -0.039765, 0.044446, -0.027757, 0.111355, 0.065736, -0.066917  # dense_weights[319]
    .float -0.006647, -0.057122, -0.182809, -0.553702, 0.394644, 0.254770, 0.080073, 0.087690, 0.071915, -0.123105  # dense_weights[320]
    .float 0.021328, -0.008088, 0.016549, 0.089916, -0.011801, 0.017561, -0.003448, 0.025614, -0.025321, -0.070584  # dense_weights[321]
    .float -0.036203, -0.000953, 0.093382, 0.086990, 0.003423, -0.091367, -0.064653, 0.085884, 0.001396, 0.009657  # dense_weights[322]
    .float 0.002852, -0.030307, -0.095320, 0.036793, 0.015631, -0.086000, 0.022324, -0.036417, -0.081100, 0.125525  # dense_weights[323]
    .float -0.134657, -0.135019, -0.002367, -0.017895, -0.053309, -0.039015, -0.041926, 0.030572, 0.112627, 0.149941  # dense_weights[324]
    .float -0.028961, -0.042604, 0.020786, 0.040798, -0.095456, -0.074288, -0.157366, 0.163661, 0.049183, -0.074052  # dense_weights[325]
    .float -0.014742, -0.099710, -0.124114, -0.148389, 0.165888, -0.027342, 0.070619, -0.132971, -0.022607, 0.149625  # dense_weights[326]
    .float -0.038146, -0.052698, -0.100446, -0.113605, 0.055272, 0.037222, 0.070858, 0.070234, 0.021787, 0.074830  # dense_weights[327]
    .float -0.007519, 0.074961, -0.005477, -0.430004, 0.262522, 0.168237, 0.030818, 0.020056, 0.059342, -0.184140  # dense_weights[328]
    .float -0.069014, 0.073107, -0.027425, 0.081401, -0.145781, -0.069193, -0.123155, 0.141654, -0.044528, -0.000782  # dense_weights[329]
    .float -0.016324, 0.040450, 0.043390, 0.133953, -0.160269, -0.046164, -0.038194, 0.079734, 0.021395, -0.053351  # dense_weights[330]
    .float -0.071475, -0.048032, -0.075326, 0.105978, 0.131167, -0.124532, -0.001210, -0.149572, -0.037234, 0.139900  # dense_weights[331]
    .float -0.027081, -0.026316, -0.079461, 0.007267, -0.085847, -0.029114, -0.048578, -0.012672, 0.196571, 0.182511  # dense_weights[332]
    .float 0.005827, 0.059674, 0.031494, 0.032547, -0.178315, -0.012583, -0.161350, 0.022434, 0.015346, 0.042774  # dense_weights[333]
    .float 0.015008, -0.028845, -0.183385, -0.138754, 0.306252, 0.063508, 0.146697, -0.156605, -0.069117, 0.048668  # dense_weights[334]
    .float 0.080597, 0.041306, -0.012602, -0.071008, -0.041853, -0.023361, 0.016757, 0.103715, -0.039938, 0.079377  # dense_weights[335]
    .float -0.058398, 0.229466, 0.005198, -0.277291, 0.216543, -0.121967, -0.063558, -0.009126, -0.039130, -0.215737  # dense_weights[336]
    .float 0.001381, -0.004715, -0.050270, -0.036143, -0.080219, -0.004640, -0.213969, 0.047012, 0.013413, 0.012348  # dense_weights[337]
    .float -0.096063, 0.011981, -0.008729, 0.093742, -0.228650, -0.068713, -0.131519, 0.071944, -0.022003, 0.041285  # dense_weights[338]
    .float 0.003506, 0.017945, -0.158909, 0.092196, 0.043082, -0.251434, -0.006193, -0.017196, -0.057898, 0.136332  # dense_weights[339]
    .float -0.053321, -0.064092, 0.046530, -0.041031, -0.106906, -0.082332, -0.067227, -0.079298, 0.136051, 0.133909  # dense_weights[340]
    .float -0.055223, 0.032171, 0.136948, 0.134555, -0.135851, -0.030659, -0.141983, 0.137632, -0.044322, 0.043724  # dense_weights[341]
    .float 0.032326, 0.039839, -0.017653, -0.046125, 0.259973, -0.171085, 0.148017, -0.044671, -0.280016, -0.098222  # dense_weights[342]
    .float -0.008142, 0.073855, -0.140196, 0.034344, -0.082584, -0.015569, -0.077850, 0.061125, -0.029222, 0.050167  # dense_weights[343]
    .float 0.034433, 0.113277, 0.142885, -0.046939, 0.099165, -0.176022, -0.217872, 0.072650, -0.012925, -0.051888  # dense_weights[344]
    .float -0.001971, 0.005014, -0.036705, 0.008648, -0.053758, 0.038029, -0.212680, 0.043866, 0.070140, -0.002652  # dense_weights[345]
    .float 0.012165, -0.094148, -0.072912, 0.011901, -0.063247, 0.123580, -0.010482, 0.088715, -0.086987, 0.063857  # dense_weights[346]
    .float -0.027155, -0.009016, -0.072424, 0.230051, 0.127995, -0.340065, -0.114708, -0.015695, 0.069080, -0.006305  # dense_weights[347]
    .float 0.002527, -0.008773, -0.073817, 0.031038, -0.122295, -0.063162, -0.107607, 0.024744, 0.022085, 0.075820  # dense_weights[348]
    .float 0.050619, -0.241952, 0.025766, 0.009698, -0.137252, 0.108951, -0.174430, 0.096527, 0.029550, 0.024169  # dense_weights[349]
    .float -0.113218, 0.244473, 0.070574, -0.021927, 0.328171, -0.460859, -0.127099, -0.084410, -0.188715, -0.215271  # dense_weights[350]
    .float 0.047567, 0.110451, -0.098716, -0.018285, -0.062463, -0.044625, -0.070792, -0.000680, -0.033277, -0.069523  # dense_weights[351]
    .float 0.117846, 0.017724, 0.093295, -0.010240, 0.107267, -0.164039, -0.307820, 0.057843, 0.032060, 0.047386  # dense_weights[352]
    .float 0.079904, -0.109403, -0.028687, -0.094002, -0.065003, 0.128183, -0.056305, 0.087828, 0.083384, 0.055535  # dense_weights[353]
    .float -0.092506, -0.118038, -0.123423, -0.123179, -0.072926, 0.222530, 0.086203, 0.003446, -0.065949, -0.001965  # dense_weights[354]
    .float -0.101575, -0.076525, 0.028305, 0.116271, 0.066183, -0.340454, -0.102199, 0.025640, 0.072541, 0.026041  # dense_weights[355]
    .float 0.113007, -0.010329, -0.011777, 0.041517, -0.107381, 0.002500, -0.108449, 0.001453, 0.035302, 0.062290  # dense_weights[356]
    .float 0.115089, -0.215282, -0.018951, -0.046959, -0.265056, 0.106904, -0.089997, 0.015188, 0.004371, 0.010854  # dense_weights[357]
    .float -0.069375, 0.095883, 0.096066, 0.074435, 0.412027, -0.447971, -0.176802, -0.064002, -0.124256, -0.127702  # dense_weights[358]
    .float 0.031237, -0.049579, 0.013455, -0.008433, 0.031625, -0.148191, -0.103840, 0.028135, 0.029398, -0.006335  # dense_weights[359]
    .float 0.106882, -0.043067, -0.000608, 0.022756, -0.025502, -0.092564, -0.164526, 0.038048, 0.038898, -0.020577  # dense_weights[360]
    .float 0.022321, -0.060556, -0.008654, -0.018197, -0.030636, 0.117307, -0.034138, 0.034422, 0.094413, -0.016700  # dense_weights[361]
    .float -0.018698, -0.038961, -0.127479, -0.116896, -0.101050, 0.254109, 0.044867, 0.076706, -0.004348, -0.060432  # dense_weights[362]
    .float -0.088505, 0.011231, -0.031552, 0.102614, 0.175233, -0.280373, -0.091698, 0.142382, 0.087337, -0.228827  # dense_weights[363]
    .float 0.123861, -0.015623, -0.049350, 0.028768, -0.012701, -0.043053, -0.053109, 0.038134, -0.011113, -0.023874  # dense_weights[364]
    .float 0.013819, -0.162754, -0.108676, -0.048709, -0.210122, 0.215993, 0.005749, -0.101911, 0.042477, 0.061146  # dense_weights[365]
    .float -0.048650, 0.093951, 0.091140, 0.072506, 0.369177, -0.447127, -0.291243, 0.126940, -0.062248, -0.399259  # dense_weights[366]
    .float 0.106267, -0.069469, 0.070062, 0.050030, -0.004270, -0.054878, -0.060847, 0.122138, 0.025359, -0.019664  # dense_weights[367]
    .float 0.085400, -0.048137, 0.007598, 0.073622, -0.110840, -0.172403, -0.019910, 0.081987, 0.004639, 0.061439  # dense_weights[368]
    .float -0.010003, -0.075797, -0.149403, -0.068132, -0.152778, 0.239034, -0.007431, 0.011756, 0.051401, -0.038057  # dense_weights[369]
    .float -0.085275, -0.052838, -0.127493, -0.167645, -0.017508, 0.282735, 0.115116, 0.042211, 0.003292, -0.048734  # dense_weights[370]
    .float -0.050461, 0.016345, -0.073607, -0.080298, 0.018692, 0.182735, -0.081350, 0.097677, 0.044794, -0.256938  # dense_weights[371]
    .float 0.083165, 0.012740, -0.026488, -0.082997, -0.107572, 0.051574, -0.056188, -0.013700, -0.002941, 0.007094  # dense_weights[372]
    .float -0.038807, -0.146039, -0.203789, -0.061520, -0.090790, 0.076019, 0.007495, -0.142384, 0.075254, -0.080213  # dense_weights[373]
    .float -0.111812, 0.112691, 0.057266, 0.035396, 0.088363, 0.034963, -0.281062, 0.115624, -0.013911, -0.236571  # dense_weights[374]
    .float 0.117445, -0.074814, 0.022096, -0.030181, -0.099294, -0.045060, -0.032307, 0.080952, 0.062034, 0.013835  # dense_weights[375]
    .float 0.069135, 0.008704, 0.008425, 0.036364, -0.123725, -0.018248, -0.046676, -0.087257, 0.059474, 0.037776  # dense_weights[376]
    .float -0.044129, -0.170079, -0.038042, -0.108770, -0.099840, 0.299970, -0.164280, -0.116327, -0.066034, -0.129560  # dense_weights[377]
    .float -0.011307, -0.234235, -0.080049, -0.118114, -0.109981, 0.418528, -0.073279, -0.055985, -0.040073, -0.060658  # dense_weights[378]
    .float -0.202193, 0.031820, -0.080165, -0.045632, -0.187251, 0.463816, -0.204002, -0.082319, -0.068738, -0.222523  # dense_weights[379]
    .float 0.069183, -0.047642, -0.147947, -0.041644, -0.016334, 0.062610, -0.048645, 0.001919, 0.077924, -0.008522  # dense_weights[380]
    .float -0.075235, -0.148240, -0.125947, -0.142129, 0.026095, 0.147321, -0.015037, -0.123906, 0.036766, -0.141575  # dense_weights[381]
    .float -0.087464, 0.044086, 0.104987, -0.050138, -0.097590, 0.079703, -0.091027, 0.005783, -0.075813, -0.014500  # dense_weights[382]
    .float 0.085820, -0.109854, -0.054426, -0.101829, -0.040012, 0.018488, -0.008147, -0.049567, 0.052527, -0.025306  # dense_weights[383]
    .float -0.078916, -0.055682, 0.009216, 0.037916, -0.035914, 0.126441, -0.052177, 0.066445, -0.071408, -0.085909  # dense_weights[384]
    .float -0.048747, -0.063469, -0.092586, -0.055435, -0.049660, -0.048509, -0.068627, 0.088378, -0.066266, -0.059988  # dense_weights[385]
    .float -0.137843, -0.058869, -0.012890, 0.076158, -0.139650, -0.074560, -0.147724, 0.133906, 0.001677, -0.084450  # dense_weights[386]
    .float -0.113541, -0.044234, 0.015508, -0.060494, -0.046100, 0.016547, -0.106147, -0.005949, -0.023168, 0.014415  # dense_weights[387]
    .float -0.055327, -0.038788, 0.073239, 0.075668, -0.084053, -0.180396, -0.210098, 0.237728, -0.081563, -0.039296  # dense_weights[388]
    .float -0.039047, -0.030917, -0.081724, -0.020864, -0.088122, -0.067455, 0.001653, 0.036420, 0.084792, -0.010710  # dense_weights[389]
    .float -0.085022, 0.029929, 0.020059, -0.050920, -0.107361, 0.040632, -0.043128, 0.002593, 0.022375, 0.053175  # dense_weights[390]
    .float -0.048068, -0.091653, -0.020548, 0.122402, -0.078853, -0.238331, -0.145841, 0.079990, 0.017826, -0.113673  # dense_weights[391]
    .float -0.070399, 0.094870, -0.066182, -0.089784, -0.088581, 0.152178, 0.010692, 0.024890, -0.113989, -0.005616  # dense_weights[392]
    .float -0.014488, -0.079387, -0.074262, -0.096124, -0.013242, -0.020577, 0.026809, 0.014587, -0.057859, -0.045271  # dense_weights[393]
    .float 0.009128, -0.057244, -0.046050, -0.000155, 0.001447, -0.121996, -0.000892, 0.071796, -0.094799, -0.033590  # dense_weights[394]
    .float 0.071652, 0.054943, -0.058036, -0.096699, -0.003289, 0.072657, 0.054280, -0.117314, -0.061132, 0.018301  # dense_weights[395]
    .float 0.004400, 0.039876, 0.159261, 0.071206, 0.068617, -0.176969, -0.048085, 0.076091, 0.008166, -0.111485  # dense_weights[396]
    .float 0.022928, 0.007005, -0.022947, -0.021869, 0.000879, 0.017487, -0.053416, -0.066767, 0.075936, 0.068494  # dense_weights[397]
    .float 0.055043, -0.074835, -0.143809, -0.064513, 0.035923, 0.070304, 0.063280, -0.026220, -0.003836, 0.102906  # dense_weights[398]
    .float 0.030587, -0.145554, 0.040026, 0.043183, -0.012847, -0.065012, -0.005605, 0.007884, 0.046372, -0.023730  # dense_weights[399]
    .float -0.049119, 0.036380, -0.202244, -0.327452, 0.101742, 0.120479, 0.157371, -0.120834, 0.202196, 0.035051  # dense_weights[400]
    .float -0.017269, -0.044001, 0.039359, 0.046440, 0.064014, 0.026600, 0.052490, 0.100207, -0.054602, 0.048004  # dense_weights[401]
    .float 0.008671, -0.027797, 0.097382, 0.039388, 0.043936, -0.078012, -0.005016, 0.050407, -0.045618, -0.061264  # dense_weights[402]
    .float 0.059329, 0.043217, -0.161509, -0.035711, 0.059661, -0.004433, 0.081423, 0.001765, -0.070552, 0.025529  # dense_weights[403]
    .float -0.051173, 0.100733, 0.008816, 0.002145, -0.003537, -0.045872, -0.096734, 0.081331, 0.052725, -0.062116  # dense_weights[404]
    .float 0.014916, -0.080545, -0.016121, 0.065720, 0.044235, 0.039514, -0.054280, 0.075579, -0.030991, 0.025405  # dense_weights[405]
    .float 0.081684, -0.009836, -0.072478, -0.063483, 0.086207, 0.051485, 0.158390, -0.194300, -0.001035, 0.027966  # dense_weights[406]
    .float 0.047028, 0.006599, -0.037675, -0.065591, 0.050101, 0.010219, -0.021369, 0.012499, -0.053919, 0.086763  # dense_weights[407]
    .float -0.041930, -0.266240, -0.304596, -0.455402, 0.094862, 0.176365, 0.128287, -0.151026, 0.173000, 0.153699  # dense_weights[408]
    .float 0.015251, -0.068343, -0.016885, 0.021342, 0.120585, 0.044448, -0.023711, 0.033739, -0.006056, 0.008360  # dense_weights[409]
    .float -0.032021, -0.043790, 0.133095, 0.107382, -0.026606, -0.073970, -0.046928, 0.145590, -0.162202, -0.080927  # dense_weights[410]
    .float 0.029384, 0.015102, -0.103454, -0.021203, 0.047289, -0.063942, 0.123263, -0.007846, -0.094757, 0.000662  # dense_weights[411]
    .float -0.040176, -0.063386, -0.037189, 0.032833, 0.077294, -0.015916, -0.029275, -0.028290, 0.034088, -0.004942  # dense_weights[412]
    .float 0.005409, -0.016928, -0.031594, -0.040410, -0.008372, 0.104864, -0.099839, -0.024678, 0.096214, 0.092779  # dense_weights[413]
    .float 0.088990, -0.014006, -0.131342, -0.030460, 0.128986, 0.026526, 0.110922, -0.135454, -0.091903, 0.177553  # dense_weights[414]
    .float 0.039753, -0.025213, -0.063890, -0.124023, 0.060878, 0.062738, -0.037473, 0.024602, 0.006523, 0.082807  # dense_weights[415]
    .float 0.015903, -0.491339, -0.235962, -0.514623, 0.177318, 0.204141, 0.089959, -0.238403, 0.271392, 0.097937  # dense_weights[416]
    .float -0.018971, -0.122547, -0.090780, 0.073579, 0.043936, 0.004376, -0.016501, -0.003172, -0.027703, -0.013495  # dense_weights[417]
    .float -0.083029, -0.146806, 0.071305, 0.043522, 0.009786, -0.034627, 0.019249, 0.090530, -0.025892, 0.021063  # dense_weights[418]
    .float -0.027643, -0.108984, -0.124691, 0.039546, 0.095991, -0.007488, 0.054161, -0.097575, -0.161447, 0.007878  # dense_weights[419]
    .float -0.073977, -0.184931, -0.026967, -0.079968, 0.084155, -0.021588, -0.063899, -0.160588, 0.055146, 0.169344  # dense_weights[420]
    .float -0.044269, -0.060850, -0.042296, -0.001882, -0.048624, 0.033989, -0.064438, 0.002845, 0.038007, -0.041071  # dense_weights[421]
    .float 0.051364, -0.055210, -0.303939, -0.015475, 0.191590, 0.073092, 0.112493, -0.283216, -0.149043, 0.109198  # dense_weights[422]
    .float 0.047076, -0.052574, -0.036460, -0.090855, -0.021589, 0.049460, 0.053035, -0.029922, -0.040456, 0.099679  # dense_weights[423]
    .float -0.006012, -0.131948, -0.053234, -0.519266, 0.178231, 0.190184, -0.015888, -0.150237, 0.126014, -0.029660  # dense_weights[424]
    .float -0.052807, 0.124276, -0.034527, 0.066894, -0.156924, -0.023939, -0.099352, -0.006022, 0.062400, 0.095091  # dense_weights[425]
    .float -0.154228, -0.041775, 0.045770, 0.100458, -0.060321, -0.005379, 0.026547, -0.017430, -0.001881, -0.035709  # dense_weights[426]
    .float -0.091552, 0.019709, -0.146014, 0.059362, -0.042117, -0.133532, -0.056776, -0.051625, -0.048893, 0.128985  # dense_weights[427]
    .float -0.091512, -0.112662, -0.092249, -0.060402, 0.025716, 0.085350, -0.020440, -0.133318, 0.145728, 0.123317  # dense_weights[428]
    .float -0.137718, 0.097461, -0.055850, 0.012131, -0.126358, -0.082046, -0.294980, 0.123919, 0.121762, 0.036115  # dense_weights[429]
    .float -0.009549, 0.060638, -0.129380, 0.026610, 0.044491, -0.027234, 0.172039, -0.023549, -0.233507, -0.029137  # dense_weights[430]
    .float -0.007088, -0.014420, -0.119151, -0.104672, 0.035308, 0.064966, -0.037634, 0.055558, -0.037080, -0.035804  # dense_weights[431]
    .float 0.029416, 0.083716, -0.029855, -0.238664, 0.073117, 0.103088, -0.030187, -0.035823, 0.093990, -0.063940  # dense_weights[432]
    .float -0.114260, 0.044023, -0.039455, 0.027786, -0.118140, 0.088548, -0.033067, 0.008090, 0.092962, 0.079111  # dense_weights[433]
    .float -0.196503, -0.012934, -0.056081, 0.075843, -0.089025, 0.010880, 0.003380, 0.048245, -0.020921, -0.026536  # dense_weights[434]
    .float -0.053479, 0.025278, -0.033126, 0.106217, 0.052643, -0.180301, -0.059445, 0.033585, -0.004774, 0.149436  # dense_weights[435]
    .float -0.093133, -0.057726, -0.108568, 0.050024, -0.136540, -0.007093, 0.002730, -0.142012, 0.082718, 0.168410  # dense_weights[436]
    .float -0.071630, 0.053735, 0.059483, 0.094478, -0.065877, 0.029764, -0.247917, 0.088423, -0.039650, 0.017542  # dense_weights[437]
    .float -0.137994, 0.083470, 0.058520, 0.118635, 0.114066, -0.220266, 0.072196, 0.120743, -0.255251, -0.052736  # dense_weights[438]
    .float -0.046748, 0.124414, -0.154794, 0.071393, -0.066294, -0.048932, -0.099756, 0.015923, -0.031819, -0.045846  # dense_weights[439]
    .float -0.089240, 0.064370, 0.016934, -0.081725, 0.096628, -0.026167, -0.100944, 0.091840, -0.009609, 0.038598  # dense_weights[440]
    .float -0.165958, -0.029791, -0.071529, 0.046330, -0.078376, 0.051534, -0.072000, -0.033473, -0.012755, 0.082989  # dense_weights[441]
    .float -0.185641, -0.009009, -0.092914, 0.152515, -0.093455, 0.120053, 0.032900, -0.011341, 0.014535, 0.147356  # dense_weights[442]
    .float -0.066583, -0.083375, 0.029352, 0.081747, 0.021241, -0.327342, -0.003071, 0.017041, 0.133316, 0.082040  # dense_weights[443]
    .float 0.004251, 0.018052, -0.125009, 0.095523, -0.019385, -0.082201, -0.024587, 0.053602, 0.120743, 0.017680  # dense_weights[444]
    .float 0.051062, -0.103125, 0.095742, 0.073542, -0.055557, 0.011737, -0.109855, 0.158242, 0.056173, -0.034440  # dense_weights[445]
    .float -0.018853, -0.029418, 0.107067, 0.061927, 0.105337, -0.657695, -0.090540, 0.121807, -0.078369, -0.102607  # dense_weights[446]
    .float -0.015819, 0.119184, -0.025183, 0.002570, 0.062106, -0.203330, -0.173094, -0.027751, -0.032327, -0.024858  # dense_weights[447]
    .float 0.154981, 0.017575, 0.051243, -0.086839, 0.157744, -0.066227, -0.259388, 0.048711, -0.044651, 0.086223  # dense_weights[448]
    .float -0.023910, -0.116672, -0.024560, -0.031964, 0.011860, 0.111450, -0.017622, 0.031967, -0.023839, 0.123747  # dense_weights[449]
    .float -0.207309, -0.218197, -0.043940, -0.002240, -0.042144, 0.126558, 0.062312, -0.008337, 0.018373, 0.062829  # dense_weights[450]
    .float -0.055657, -0.217785, 0.034020, 0.083315, 0.034420, -0.494516, -0.114301, 0.031515, 0.101143, -0.022653  # dense_weights[451]
    .float 0.071504, -0.048552, -0.057765, 0.026113, 0.065937, -0.074803, -0.151619, -0.004672, 0.037512, 0.072563  # dense_weights[452]
    .float 0.053173, -0.172832, -0.027416, -0.026372, -0.166315, 0.004457, -0.078152, 0.101723, 0.010276, -0.036952  # dense_weights[453]
    .float -0.044847, -0.038883, 0.154754, 0.080687, 0.213667, -0.692651, -0.275589, 0.050134, 0.075632, 0.072007  # dense_weights[454]
    .float 0.024006, -0.021084, -0.058507, 0.020792, 0.057821, -0.178293, -0.124690, 0.044896, 0.070003, 0.018678  # dense_weights[455]
    .float 0.074952, -0.023198, 0.026327, 0.035370, -0.025613, -0.173986, -0.191102, -0.051479, -0.049319, 0.071501  # dense_weights[456]
    .float -0.068099, -0.063453, -0.093040, -0.149137, -0.014367, 0.125769, 0.064593, -0.035465, 0.043941, -0.009941  # dense_weights[457]
    .float -0.076196, -0.119818, -0.092287, -0.104079, -0.071692, 0.196730, 0.193076, -0.002828, -0.021079, 0.000041  # dense_weights[458]
    .float -0.035185, -0.086463, 0.058958, 0.111305, 0.072669, -0.420618, -0.019196, 0.023810, 0.183728, -0.006468  # dense_weights[459]
    .float 0.059527, -0.002698, 0.043603, -0.039560, 0.073386, -0.126060, -0.091541, 0.034782, 0.066975, 0.104035  # dense_weights[460]
    .float 0.031539, -0.088017, -0.044826, 0.002215, -0.151536, 0.032082, -0.018481, -0.094337, 0.138468, -0.066705  # dense_weights[461]
    .float 0.029942, 0.191286, 0.166356, -0.017602, 0.157330, -0.659259, -0.321187, 0.068948, 0.205301, 0.011322  # dense_weights[462]
    .float 0.027234, 0.021826, 0.038093, 0.063226, 0.014674, -0.051400, -0.164268, 0.063000, 0.081487, 0.034602  # dense_weights[463]
    .float 0.086998, -0.015010, -0.023617, -0.003463, 0.026299, -0.303397, 0.082092, 0.064436, -0.027771, 0.118244  # dense_weights[464]
    .float -0.002094, -0.031714, -0.111915, -0.140073, -0.037471, 0.183370, 0.087856, -0.126117, -0.034031, 0.021855  # dense_weights[465]
    .float -0.127472, -0.072041, -0.132081, -0.136096, -0.054035, 0.162976, 0.059961, -0.042175, -0.009864, -0.201207  # dense_weights[466]
    .float -0.005439, 0.007195, 0.079388, -0.086105, 0.080416, -0.275182, -0.033394, 0.019559, 0.274269, -0.207579  # dense_weights[467]
    .float -0.014159, -0.076297, 0.061571, -0.013508, -0.045614, -0.160908, -0.023998, -0.072254, 0.014192, 0.027195  # dense_weights[468]
    .float -0.062153, -0.130100, -0.073588, 0.061151, -0.131348, 0.135366, -0.004186, -0.036404, 0.112067, -0.107557  # dense_weights[469]
    .float 0.207185, 0.065766, 0.216742, 0.048876, -0.077591, -0.404278, -0.296000, -0.104185, 0.194502, -0.060655  # dense_weights[470]
    .float 0.106768, -0.081366, 0.017870, 0.036382, -0.059749, -0.132327, -0.053204, -0.034806, 0.049374, 0.076010  # dense_weights[471]
    .float 0.121572, 0.025355, 0.007757, -0.047222, -0.114678, -0.134234, 0.050428, -0.093993, 0.040416, 0.062118  # dense_weights[472]
    .float -0.052487, -0.017459, -0.060277, -0.171807, -0.099345, 0.197803, 0.020617, -0.105122, -0.000113, -0.048919  # dense_weights[473]
    .float -0.041713, 0.036779, -0.075761, -0.158026, -0.081547, 0.168527, 0.168346, -0.051909, -0.132998, -0.068872  # dense_weights[474]
    .float -0.099074, 0.055446, 0.210726, -0.101549, -0.026312, 0.097556, -0.103943, -0.056430, -0.110373, -0.132813  # dense_weights[475]
    .float 0.016214, -0.019275, -0.030982, -0.012687, -0.082944, 0.102383, -0.073960, -0.103885, 0.021524, -0.063287  # dense_weights[476]
    .float -0.158743, -0.157239, -0.053476, -0.149177, -0.158463, 0.291553, -0.236351, -0.129034, 0.020130, -0.145301  # dense_weights[477]
    .float -0.127516, 0.054148, 0.189774, 0.005367, -0.084875, 0.092023, -0.244569, -0.031045, -0.172735, -0.047077  # dense_weights[478]
    .float 0.002827, -0.102519, -0.129132, -0.108452, -0.112909, 0.098570, 0.028526, -0.144684, 0.114810, -0.044124  # dense_weights[479]
    .float 0.009501, 0.064766, 0.053311, -0.030744, -0.120918, 0.134451, -0.067119, -0.040586, -0.127118, 0.034040  # dense_weights[480]
    .float 0.075677, -0.079579, 0.037569, -0.099583, 0.052503, -0.099146, -0.050277, 0.042976, -0.044968, -0.010515  # dense_weights[481]
    .float -0.091691, -0.018145, 0.079374, -0.013347, -0.062314, -0.115185, -0.177590, 0.083550, -0.101501, -0.039171  # dense_weights[482]
    .float -0.056472, -0.024690, -0.104504, -0.125832, 0.077434, -0.068547, -0.010349, -0.014332, -0.049253, 0.029444  # dense_weights[483]
    .float -0.076589, 0.085427, 0.126109, 0.117324, -0.067377, -0.141750, -0.258992, 0.093732, -0.119143, -0.130595  # dense_weights[484]
    .float 0.057688, 0.064744, -0.114825, -0.145152, -0.011611, -0.106564, -0.057788, 0.164638, 0.051375, 0.019896  # dense_weights[485]
    .float -0.051242, 0.059541, -0.070174, -0.178620, -0.047345, 0.030915, 0.007955, -0.000421, -0.029167, 0.089404  # dense_weights[486]
    .float 0.025964, -0.016128, -0.026062, 0.026434, -0.027593, -0.080511, -0.107515, 0.147253, 0.027393, -0.092211  # dense_weights[487]
    .float -0.196913, 0.112364, -0.200898, 0.040849, -0.022803, 0.214213, -0.042443, -0.177557, -0.135800, 0.229869  # dense_weights[488]
    .float 0.070233, 0.037690, 0.047824, -0.079087, 0.030273, -0.020133, 0.049591, 0.124458, -0.006209, -0.106126  # dense_weights[489]
    .float 0.027524, -0.044145, 0.107550, -0.019645, 0.021352, -0.119754, -0.026256, 0.039013, 0.012315, -0.134135  # dense_weights[490]
    .float 0.094970, -0.060242, -0.095888, -0.077538, 0.115082, -0.083125, 0.034738, 0.035349, -0.083236, 0.069233  # dense_weights[491]
    .float -0.098690, 0.017009, 0.235419, 0.013207, -0.012973, -0.086591, -0.054510, 0.019332, 0.084673, -0.010223  # dense_weights[492]
    .float 0.008882, -0.003206, -0.069836, -0.084036, -0.016387, 0.019824, 0.065132, 0.103185, -0.049940, -0.052576  # dense_weights[493]
    .float 0.070098, -0.084478, -0.023196, -0.181559, -0.002088, -0.028712, 0.069849, -0.050259, -0.000258, 0.029283  # dense_weights[494]
    .float -0.039354, -0.023315, -0.006435, -0.153960, 0.080672, -0.013439, 0.047121, 0.089035, -0.020730, 0.022874  # dense_weights[495]
    .float 0.033968, -0.102449, -0.340873, -0.044007, 0.140106, 0.020962, 0.100791, -0.264652, 0.132992, 0.272279  # dense_weights[496]
    .float -0.002593, 0.043376, 0.093745, -0.037333, 0.122471, 0.009383, 0.078802, 0.063131, 0.030670, -0.031430  # dense_weights[497]
    .float 0.003774, 0.087142, 0.037124, 0.008415, 0.024025, -0.064128, -0.074604, 0.121354, 0.003003, -0.115241  # dense_weights[498]
    .float 0.002462, -0.038674, 0.053013, 0.010217, 0.124959, -0.084010, 0.067598, 0.033518, -0.149787, -0.029604  # dense_weights[499]
    .float 0.071872, 0.076362, 0.065151, -0.002182, -0.013178, -0.036863, -0.126233, -0.070455, 0.126838, -0.003300  # dense_weights[500]
    .float -0.031879, 0.005847, -0.072805, 0.022711, 0.029751, 0.136956, 0.075800, -0.072815, -0.003379, 0.043840  # dense_weights[501]
    .float 0.103416, 0.068948, -0.013314, 0.010143, 0.060825, -0.046750, 0.137855, -0.094715, -0.152799, -0.076329  # dense_weights[502]
    .float 0.021690, -0.080217, -0.077917, -0.063405, 0.047326, 0.036922, -0.016560, -0.002313, 0.016758, -0.015933  # dense_weights[503]
    .float 0.064200, -0.083931, -0.376443, -0.154946, 0.114684, 0.058886, 0.117413, -0.403244, 0.146933, 0.197477  # dense_weights[504]
    .float -0.008683, -0.016866, 0.066380, -0.018499, 0.060137, 0.000890, 0.063320, 0.052287, -0.003600, -0.011706  # dense_weights[505]
    .float 0.067284, -0.014080, 0.081125, -0.016598, 0.058129, -0.007330, 0.020284, 0.122623, -0.052989, -0.139442  # dense_weights[506]
    .float 0.086927, 0.041900, 0.027406, -0.008397, 0.000049, 0.001589, 0.151753, 0.047366, -0.033366, -0.023309  # dense_weights[507]
    .float 0.020880, -0.025524, 0.018315, -0.138343, -0.032584, 0.050320, -0.127458, -0.180173, 0.099468, 0.000205  # dense_weights[508]
    .float -0.057037, -0.109806, -0.100237, -0.000153, 0.009860, 0.054993, -0.069815, -0.008764, -0.036272, 0.008511  # dense_weights[509]
    .float 0.049010, -0.037402, -0.135599, 0.142552, 0.097336, -0.012221, 0.084422, -0.023965, -0.231418, 0.002026  # dense_weights[510]
    .float 0.117546, -0.007564, -0.058092, -0.103684, 0.086477, -0.020337, -0.004246, -0.081010, 0.008311, -0.034893  # dense_weights[511]
    .float 0.000125, -0.233779, -0.346053, -0.419881, 0.069195, 0.193708, -0.048328, -0.121195, 0.254483, 0.139670  # dense_weights[512]
    .float -0.030104, -0.004910, 0.075436, 0.135920, 0.056415, -0.008531, 0.031306, -0.035688, -0.024588, 0.002384  # dense_weights[513]
    .float -0.022891, -0.116161, 0.165900, 0.085816, 0.000577, 0.016263, -0.054334, -0.032363, -0.067162, -0.072549  # dense_weights[514]
    .float 0.042953, -0.084240, 0.029962, 0.086923, 0.032334, -0.075685, 0.022391, -0.027152, 0.018458, -0.045398  # dense_weights[515]
    .float -0.058693, -0.303448, -0.021671, -0.177435, 0.109456, 0.051178, 0.002234, -0.165512, 0.087435, 0.089791  # dense_weights[516]
    .float -0.141921, -0.013716, -0.045288, 0.072148, -0.012009, 0.053568, -0.156413, -0.005398, 0.098208, -0.009895  # dense_weights[517]
    .float 0.081859, 0.005808, -0.183317, 0.091550, 0.082505, 0.069057, 0.139351, -0.061782, -0.259730, 0.015238  # dense_weights[518]
    .float 0.076128, -0.036724, -0.114021, -0.103980, 0.064786, 0.025331, 0.099472, -0.031816, -0.016975, -0.007297  # dense_weights[519]
    .float 0.088718, -0.111255, -0.203400, -0.378762, 0.085168, 0.171353, -0.022283, -0.147088, 0.121595, -0.089707  # dense_weights[520]
    .float -0.072323, 0.047732, 0.042801, 0.104107, -0.055566, 0.028726, -0.041620, -0.203711, 0.001085, 0.124233  # dense_weights[521]
    .float -0.097986, -0.053915, 0.113114, 0.128213, 0.039097, 0.069169, -0.010048, -0.101241, -0.043058, 0.052919  # dense_weights[522]
    .float -0.108558, 0.021318, -0.005067, 0.110673, -0.110646, -0.018757, -0.101311, -0.093874, 0.068154, 0.078329  # dense_weights[523]
    .float -0.074328, -0.071036, -0.091429, -0.033087, 0.078258, 0.114089, 0.104278, -0.180328, 0.063050, 0.081962  # dense_weights[524]
    .float -0.227705, 0.008428, -0.006587, 0.044034, -0.077688, 0.109902, -0.145130, -0.026417, 0.163634, 0.047502  # dense_weights[525]
    .float -0.033392, 0.094466, -0.071480, 0.186340, -0.184866, -0.069386, 0.054436, 0.040306, -0.242803, -0.253279  # dense_weights[526]
    .float -0.041101, 0.115463, -0.199638, 0.043975, 0.094507, 0.068688, 0.094436, -0.072339, 0.063329, -0.075415  # dense_weights[527]
    .float 0.088951, 0.132012, 0.045568, -0.171533, -0.037032, 0.082794, -0.013622, -0.097866, 0.022379, -0.201481  # dense_weights[528]
    .float -0.231132, 0.040819, -0.043478, 0.031433, 0.050670, -0.034986, 0.021503, -0.167128, 0.048224, 0.156599  # dense_weights[529]
    .float -0.154010, 0.061262, 0.099377, 0.145164, 0.017017, 0.127305, -0.062671, -0.088769, -0.086752, 0.117197  # dense_weights[530]
    .float -0.097516, -0.008970, 0.036935, -0.048732, -0.036045, -0.228775, -0.038324, 0.070153, 0.045598, 0.138647  # dense_weights[531]
    .float -0.097828, -0.064991, -0.120168, 0.038581, -0.100159, 0.065001, 0.121945, -0.175269, 0.154584, 0.025246  # dense_weights[532]
    .float -0.146016, 0.095325, -0.036117, 0.089601, 0.012457, 0.060535, -0.158127, -0.051918, -0.044199, 0.115810  # dense_weights[533]
    .float -0.084712, 0.059760, 0.018311, 0.025312, -0.088725, -0.259151, 0.105435, 0.272777, -0.284721, -0.157053  # dense_weights[534]
    .float -0.103646, 0.016697, -0.108026, 0.125581, -0.014000, -0.111801, 0.014348, -0.044345, 0.075735, -0.024200  # dense_weights[535]
    .float -0.040345, 0.051983, -0.096468, -0.054536, 0.106947, 0.069915, -0.158125, 0.059145, -0.033095, 0.099023  # dense_weights[536]
    .float -0.203010, -0.043778, -0.060635, 0.033212, 0.023902, 0.041828, 0.066759, -0.094436, 0.038674, 0.070383  # dense_weights[537]
    .float -0.166365, -0.167439, -0.017335, 0.147580, -0.046341, 0.008989, 0.055505, -0.000368, -0.075629, 0.056106  # dense_weights[538]
    .float -0.039795, -0.076627, 0.107775, 0.028964, 0.021203, -0.162439, -0.051425, 0.022491, -0.031908, -0.023837  # dense_weights[539]
    .float -0.073584, 0.050779, -0.054184, 0.053330, -0.016892, -0.095404, -0.021758, 0.051601, 0.145003, -0.017331  # dense_weights[540]
    .float -0.005880, -0.122237, -0.107088, 0.115413, -0.026530, 0.032844, -0.031751, 0.006414, -0.036371, 0.182710  # dense_weights[541]
    .float 0.100209, 0.078099, 0.126852, -0.095859, -0.030112, -0.474633, -0.148652, 0.175603, -0.092275, -0.132786  # dense_weights[542]
    .float -0.041908, 0.057584, -0.067193, 0.022668, -0.003475, -0.135123, -0.062521, 0.052790, 0.033350, 0.173509  # dense_weights[543]
    .float 0.065728, 0.060070, -0.007630, 0.026305, 0.130862, 0.037935, -0.114995, 0.066724, -0.018372, 0.082279  # dense_weights[544]
    .float -0.166491, -0.165419, -0.085931, 0.035278, -0.040759, 0.002211, 0.087760, 0.001949, -0.101520, 0.108215  # dense_weights[545]
    .float -0.200500, -0.265541, -0.008866, 0.062048, 0.028558, 0.041339, 0.131093, 0.038515, 0.008139, 0.015620  # dense_weights[546]
    .float 0.042836, -0.178945, 0.244953, -0.052918, 0.068553, -0.277196, -0.060573, -0.059703, 0.031787, 0.086122  # dense_weights[547]
    .float -0.131045, -0.052551, -0.035952, -0.021400, 0.105187, -0.066628, -0.086544, 0.108454, 0.122239, 0.006008  # dense_weights[548]
    .float -0.003362, -0.203147, -0.093709, 0.063439, -0.024293, -0.046031, -0.019100, 0.071097, 0.147777, -0.009527  # dense_weights[549]
    .float 0.097481, 0.167566, 0.290994, -0.307974, 0.097533, -0.467081, -0.335893, 0.057441, 0.079266, 0.075277  # dense_weights[550]
    .float 0.030953, -0.000296, -0.016559, 0.065835, 0.014628, -0.114969, -0.070923, 0.005196, 0.001219, 0.098521  # dense_weights[551]
    .float 0.017419, 0.019031, -0.051386, -0.025814, -0.058396, -0.020162, -0.033348, 0.009377, -0.107384, 0.039391  # dense_weights[552]
    .float -0.045157, -0.138171, -0.115123, 0.030336, 0.066366, 0.128051, 0.112830, -0.040344, -0.038139, 0.015458  # dense_weights[553]
    .float -0.093118, -0.224560, -0.087215, 0.057194, 0.012094, 0.012129, 0.044418, 0.011317, 0.027856, -0.182082  # dense_weights[554]
    .float 0.003577, -0.047375, 0.079276, -0.068480, 0.082286, -0.377534, 0.013160, -0.026568, 0.066530, 0.076168  # dense_weights[555]
    .float 0.003223, 0.032255, -0.044458, -0.019632, -0.022175, -0.039503, -0.089658, 0.027553, 0.046446, 0.062739  # dense_weights[556]
    .float -0.051443, -0.091844, -0.039018, 0.073504, -0.010796, -0.093348, 0.026915, -0.039539, 0.085729, -0.042693  # dense_weights[557]
    .float 0.163107, 0.129366, 0.194746, -0.352219, 0.074733, -0.545613, -0.186882, -0.047334, 0.010329, 0.201686  # dense_weights[558]
    .float 0.090442, -0.076375, -0.040611, 0.064871, 0.047960, -0.069171, -0.054655, 0.076572, 0.013253, 0.028012  # dense_weights[559]
    .float 0.127088, -0.017008, -0.018801, -0.034160, 0.090559, -0.029700, 0.025168, 0.044584, -0.087236, 0.079795  # dense_weights[560]
    .float 0.013779, -0.027667, -0.080541, -0.126030, 0.037462, 0.137769, 0.071810, 0.024641, 0.001715, -0.086551  # dense_weights[561]
    .float -0.119123, -0.179591, -0.013000, 0.037494, 0.035696, 0.131630, 0.131845, 0.101937, 0.058246, -0.179659  # dense_weights[562]
    .float 0.130422, -0.111672, 0.093850, 0.003294, 0.021368, -0.277968, -0.026780, 0.098979, -0.034097, -0.186348  # dense_weights[563]
    .float 0.082225, -0.037204, 0.021171, -0.091843, -0.049357, -0.050770, 0.028635, 0.063377, 0.014068, -0.031632  # dense_weights[564]
    .float 0.058117, -0.047945, -0.059714, -0.046402, 0.057108, 0.022407, 0.067185, -0.091784, 0.142096, -0.189614  # dense_weights[565]
    .float 0.270963, 0.047131, 0.269043, -0.082034, -0.095546, -0.368134, -0.077563, -0.098067, -0.062167, 0.122563  # dense_weights[566]
    .float 0.044430, -0.096443, -0.104031, -0.043027, -0.055171, -0.096310, 0.079720, -0.055229, 0.048815, 0.037602  # dense_weights[567]
    .float 0.121396, -0.018313, -0.113494, -0.100615, 0.022693, -0.124689, 0.063968, 0.034710, 0.047380, 0.047681  # dense_weights[568]
    .float -0.057760, -0.051296, 0.149569, -0.167034, 0.012447, 0.078171, -0.010167, 0.029014, -0.101917, -0.102228  # dense_weights[569]
    .float -0.071197, -0.110994, 0.113787, -0.044205, 0.032687, 0.105975, 0.037586, 0.098628, -0.148742, -0.268901  # dense_weights[570]
    .float -0.077545, -0.047451, 0.273008, 0.041830, -0.099756, -0.008207, -0.293775, 0.051352, -0.165635, -0.164453  # dense_weights[571]
    .float -0.009917, -0.000423, -0.007534, -0.164354, -0.069102, -0.072241, 0.084154, -0.005991, 0.038983, -0.019925  # dense_weights[572]
    .float -0.099724, 0.037909, 0.100260, -0.211284, -0.089061, 0.150638, -0.115175, -0.178620, 0.104909, -0.070417  # dense_weights[573]
    .float -0.043616, 0.016672, 0.283437, 0.065011, -0.108608, 0.073875, -0.239717, 0.017461, -0.150491, 0.052807  # dense_weights[574]
    .float 0.073241, -0.027817, -0.087013, -0.096544, -0.034666, 0.017355, 0.011662, -0.047463, -0.050314, -0.077832  # dense_weights[575]
    .float -0.103480, 0.058978, -0.035041, 0.026321, -0.104204, 0.082321, -0.014214, 0.061771, -0.147144, 0.036673  # dense_weights[576]
    .float 0.063743, -0.099437, 0.143372, -0.033053, -0.043268, 0.022603, -0.068684, -0.009508, -0.064379, -0.061562  # dense_weights[577]
    .float -0.042363, -0.017438, 0.128067, 0.032465, -0.068603, -0.026824, -0.104435, 0.117805, -0.059861, -0.042356  # dense_weights[578]
    .float 0.053483, 0.046332, 0.110327, -0.098828, -0.021492, -0.129403, -0.019745, 0.055816, -0.146686, 0.029624  # dense_weights[579]
    .float -0.063114, 0.017001, 0.072182, 0.089848, -0.184186, 0.047108, -0.288758, -0.064128, 0.005305, -0.074706  # dense_weights[580]
    .float 0.005009, 0.085381, -0.048767, -0.024906, 0.041914, -0.135789, -0.081511, 0.045474, -0.105729, -0.093485  # dense_weights[581]
    .float 0.099704, -0.016039, 0.076348, -0.068205, 0.049042, -0.139558, 0.023480, 0.038016, -0.040273, 0.050465  # dense_weights[582]
    .float 0.003893, 0.064815, -0.036024, 0.017464, -0.003681, -0.004262, -0.108962, 0.084091, 0.006898, -0.088653  # dense_weights[583]
    .float -0.089177, -0.032074, -0.159903, 0.160847, -0.099016, 0.205439, -0.161820, -0.081340, -0.228082, 0.279126  # dense_weights[584]
    .float -0.021959, -0.127749, 0.089038, -0.152686, 0.060301, -0.069763, 0.000994, 0.052860, 0.032509, -0.100122  # dense_weights[585]
    .float -0.074708, -0.147080, 0.075569, -0.034198, 0.037622, 0.073884, -0.024099, 0.067238, -0.002124, -0.036339  # dense_weights[586]
    .float 0.025563, -0.120561, 0.078157, -0.150809, 0.097293, -0.102226, 0.030132, 0.059212, -0.013987, -0.104142  # dense_weights[587]
    .float -0.090583, -0.130060, 0.103085, -0.005564, -0.100496, 0.127452, -0.036945, -0.005712, 0.052212, 0.013245  # dense_weights[588]
    .float 0.035277, -0.031336, -0.078980, -0.117666, 0.043103, -0.058412, 0.041379, 0.003054, 0.019158, -0.006664  # dense_weights[589]
    .float 0.137793, -0.032589, 0.027861, -0.135012, 0.000927, -0.036487, -0.001201, 0.007769, -0.056920, -0.054697  # dense_weights[590]
    .float 0.120496, -0.018492, -0.027730, -0.096890, 0.020958, -0.131248, -0.001614, 0.050172, 0.000479, -0.071379  # dense_weights[591]
    .float 0.078579, -0.161497, -0.354678, 0.224064, -0.034418, 0.117986, 0.141642, -0.207261, -0.314663, 0.239441  # dense_weights[592]
    .float 0.083995, -0.026621, 0.100613, -0.058776, 0.082116, -0.052839, 0.073748, 0.013879, 0.062049, -0.107839  # dense_weights[593]
    .float 0.001401, -0.041451, 0.158547, -0.041727, 0.041778, -0.049903, -0.085762, 0.030534, 0.066691, -0.157124  # dense_weights[594]
    .float 0.115446, 0.067799, 0.011350, -0.108604, 0.041664, -0.162279, 0.076141, 0.143638, 0.030950, -0.167163  # dense_weights[595]
    .float -0.046924, -0.014963, 0.022426, -0.072760, -0.010523, 0.026712, -0.064386, 0.016952, 0.145368, 0.070817  # dense_weights[596]
    .float 0.023143, -0.005592, -0.170296, -0.107402, 0.094234, 0.053767, -0.005090, 0.052127, -0.084014, 0.045170  # dense_weights[597]
    .float 0.114588, -0.073267, 0.092308, -0.092070, 0.058889, -0.096141, 0.028860, 0.109062, -0.019292, -0.179760  # dense_weights[598]
    .float 0.087036, 0.000059, -0.031370, -0.094900, 0.066622, -0.060147, -0.012909, -0.024357, 0.061450, 0.020552  # dense_weights[599]
    .float 0.075330, -0.192284, -0.512653, 0.194269, 0.029300, 0.018636, 0.124502, -0.106914, -0.178265, 0.269397  # dense_weights[600]
    .float -0.029212, -0.069434, 0.124264, -0.135301, 0.002250, 0.007236, 0.023243, -0.035699, -0.096790, -0.052027  # dense_weights[601]
    .float -0.003886, -0.035140, 0.158140, -0.085145, 0.030394, 0.012764, -0.060502, 0.121019, -0.102487, -0.047370  # dense_weights[602]
    .float 0.118111, -0.041550, 0.137123, -0.121397, 0.004821, -0.076984, 0.123077, 0.060759, 0.067917, -0.100678  # dense_weights[603]
    .float 0.070593, -0.052759, -0.130658, -0.048229, 0.041794, -0.064336, -0.060756, 0.013141, -0.000414, 0.110492  # dense_weights[604]
    .float -0.129149, -0.064660, -0.101177, -0.089590, 0.094747, 0.176321, -0.000164, 0.049254, 0.014063, 0.024743  # dense_weights[605]
    .float 0.098894, -0.055612, 0.041427, -0.017693, -0.065259, -0.191447, 0.195127, 0.237338, -0.093905, -0.111545  # dense_weights[606]
    .float 0.065432, -0.099819, -0.084939, -0.099867, 0.094211, -0.054218, 0.149764, 0.070455, -0.030247, 0.030369  # dense_weights[607]
    .float 0.166944, -0.073604, -0.463758, -0.053512, -0.027238, 0.127971, 0.030042, 0.000832, 0.009471, 0.174486  # dense_weights[608]
    .float -0.074862, -0.051932, 0.140832, -0.055973, -0.038247, -0.057453, -0.014476, -0.064914, 0.047004, 0.045105  # dense_weights[609]
    .float -0.108816, -0.158741, 0.075953, 0.074897, 0.027526, 0.046144, -0.009767, -0.018914, -0.058882, -0.003948  # dense_weights[610]
    .float -0.056828, -0.025286, 0.156600, -0.057426, -0.083607, -0.174730, 0.108483, -0.031931, 0.064133, -0.074344  # dense_weights[611]
    .float -0.010775, -0.367997, -0.115165, -0.073121, -0.026502, 0.003640, 0.072667, -0.124454, 0.025696, 0.091521  # dense_weights[612]
    .float -0.051734, -0.057109, -0.086448, 0.102516, 0.089694, 0.174515, -0.129788, -0.034181, -0.017249, -0.004489  # dense_weights[613]
    .float 0.077712, 0.071747, 0.049582, -0.074309, -0.021404, -0.077698, 0.099422, 0.174371, -0.063545, -0.304235  # dense_weights[614]
    .float 0.059312, 0.021433, -0.054338, -0.131687, 0.086408, -0.001278, 0.036259, -0.107450, -0.002100, -0.020690  # dense_weights[615]
    .float 0.190533, -0.278797, -0.282642, -0.218291, -0.037114, 0.179565, 0.148271, -0.105669, -0.012439, -0.094219  # dense_weights[616]
    .float -0.158863, 0.022508, 0.031251, -0.020379, -0.006622, 0.069114, 0.009397, -0.188067, -0.055680, 0.049170  # dense_weights[617]
    .float -0.232489, -0.022833, 0.033819, 0.149800, -0.048215, 0.081992, -0.034997, -0.163861, -0.011976, 0.011811  # dense_weights[618]
    .float -0.160546, 0.122926, 0.111930, -0.086647, -0.089056, -0.142366, 0.023473, -0.085478, 0.030937, 0.039409  # dense_weights[619]
    .float 0.000728, -0.117733, -0.015582, 0.038688, 0.051290, 0.032171, 0.115367, -0.320998, -0.010332, 0.013797  # dense_weights[620]
    .float -0.215520, 0.007898, -0.046175, 0.109246, 0.063550, 0.039421, -0.129982, -0.261445, 0.117711, 0.058298  # dense_weights[621]
    .float -0.128295, 0.125184, 0.052129, 0.017627, -0.217610, -0.202047, 0.109303, 0.114875, -0.042405, -0.254494  # dense_weights[622]
    .float -0.040756, 0.104920, -0.001823, 0.060992, -0.045250, 0.019433, 0.049955, -0.163240, 0.016988, 0.059334  # dense_weights[623]
    .float 0.115465, 0.072090, -0.011565, -0.002428, -0.165919, 0.244142, 0.025511, -0.092783, -0.046883, -0.270858  # dense_weights[624]
    .float -0.100521, -0.067320, 0.142818, -0.013787, -0.022272, -0.021022, 0.110208, -0.023155, -0.039805, 0.035641  # dense_weights[625]
    .float -0.109175, -0.043070, 0.062547, 0.105237, 0.095822, -0.029068, -0.031084, -0.041541, -0.080514, 0.045338  # dense_weights[626]
    .float -0.015467, 0.002238, 0.077144, -0.099644, 0.043459, -0.134631, 0.025966, -0.015568, -0.026112, 0.141365  # dense_weights[627]
    .float -0.134054, -0.029285, 0.010982, 0.099854, -0.058808, -0.052499, 0.102883, -0.184410, 0.121322, 0.029565  # dense_weights[628]
    .float -0.218222, 0.037968, -0.091625, 0.060810, 0.070361, 0.014445, -0.070436, -0.119817, -0.032132, 0.248106  # dense_weights[629]
    .float 0.113423, 0.056448, 0.044903, -0.224117, -0.080571, -0.082353, 0.134086, 0.269505, -0.187545, -0.186451  # dense_weights[630]
    .float -0.146661, 0.091195, 0.009423, -0.036533, 0.090984, -0.133740, 0.032430, -0.040357, 0.080273, 0.096186  # dense_weights[631]
    .float -0.162008, 0.063032, -0.147668, -0.080682, -0.049487, 0.028202, -0.087319, 0.003523, 0.067129, 0.071844  # dense_weights[632]
    .float -0.161143, -0.229740, -0.032396, 0.094016, 0.062104, 0.023877, 0.143913, 0.062195, -0.075246, -0.031867  # dense_weights[633]
    .float -0.164706, -0.228869, 0.095254, 0.028789, -0.058173, 0.093523, 0.009316, 0.016071, -0.058875, 0.090442  # dense_weights[634]
    .float 0.050237, -0.052502, 0.138283, -0.206635, 0.036296, -0.110263, -0.003540, 0.064621, -0.010053, 0.063580  # dense_weights[635]
    .float -0.132042, 0.086045, -0.048848, 0.040017, -0.015405, -0.055783, 0.037596, 0.075326, 0.015425, 0.001317  # dense_weights[636]
    .float -0.117642, -0.095143, -0.033680, 0.050619, 0.052855, -0.072805, 0.046135, -0.014814, -0.034509, 0.135199  # dense_weights[637]
    .float 0.147648, -0.006006, 0.119822, -0.267550, 0.028721, -0.130371, -0.178104, 0.269399, -0.179349, -0.101790  # dense_weights[638]
    .float -0.017321, 0.079211, -0.000887, 0.017037, 0.034783, -0.038677, -0.016847, 0.027059, -0.027419, 0.149693  # dense_weights[639]
    .float 0.002958, -0.040299, -0.107350, 0.044214, 0.099687, -0.013496, 0.035354, -0.015053, 0.007671, 0.007124  # dense_weights[640]
    .float -0.104081, -0.217415, 0.002062, 0.020293, 0.048011, 0.013146, 0.019484, 0.107781, 0.009666, -0.040955  # dense_weights[641]
    .float -0.010370, -0.293114, 0.018965, 0.086810, 0.022394, 0.068900, 0.099197, 0.124112, 0.032569, -0.086748  # dense_weights[642]
    .float 0.121107, -0.004956, 0.105595, -0.132731, -0.058493, -0.087163, 0.035774, 0.010099, -0.247207, 0.113790  # dense_weights[643]
    .float -0.036774, -0.034818, -0.024895, 0.033671, 0.104031, -0.076712, -0.050575, 0.038046, 0.061943, 0.074037  # dense_weights[644]
    .float 0.003609, -0.105591, -0.129666, 0.020411, 0.010095, -0.085866, 0.031978, 0.115057, 0.092085, 0.027490  # dense_weights[645]
    .float 0.085350, 0.231142, 0.109106, -0.237073, 0.051359, -0.125854, -0.129746, 0.172864, -0.466777, 0.152955  # dense_weights[646]
    .float -0.013390, -0.087457, -0.053077, -0.005225, 0.051424, -0.013003, 0.007687, 0.026207, -0.028404, 0.144737  # dense_weights[647]
    .float 0.093708, -0.004525, -0.028637, 0.047989, 0.052049, 0.030026, -0.040493, 0.047921, -0.047232, 0.000510  # dense_weights[648]
    .float -0.037442, -0.077118, 0.010702, 0.012220, 0.069194, -0.023604, 0.017850, 0.077420, 0.001789, -0.156528  # dense_weights[649]
    .float -0.031749, -0.144090, 0.040719, 0.143544, 0.025857, -0.082653, 0.006251, 0.131781, 0.005058, -0.226556  # dense_weights[650]
    .float 0.136460, 0.022578, 0.016581, -0.188630, 0.020945, -0.126769, 0.002182, 0.096940, -0.253772, 0.018770  # dense_weights[651]
    .float -0.025158, -0.043269, 0.000971, -0.015834, 0.077695, -0.063581, 0.022688, -0.010625, 0.058320, 0.044454  # dense_weights[652]
    .float -0.027998, -0.192778, -0.089054, -0.033769, 0.155024, -0.007821, 0.062939, 0.138470, 0.112499, -0.069364  # dense_weights[653]
    .float 0.187463, 0.092217, 0.100454, -0.298087, 0.045159, -0.137944, 0.030234, 0.092893, -0.494696, 0.134068  # dense_weights[654]
    .float 0.061569, -0.122307, 0.000354, -0.050808, 0.024847, -0.091161, 0.100155, 0.091894, -0.079792, 0.002838  # dense_weights[655]
    .float -0.020730, 0.022025, -0.188750, -0.062419, 0.087416, -0.074693, -0.005535, -0.003638, -0.073578, 0.014497  # dense_weights[656]
    .float 0.005745, -0.114455, 0.056201, -0.005130, 0.060509, 0.045612, 0.017741, 0.118663, -0.030944, -0.144140  # dense_weights[657]
    .float -0.102491, -0.027455, 0.056781, 0.061178, 0.079301, -0.067881, -0.073682, 0.009297, -0.019021, -0.255615  # dense_weights[658]
    .float 0.041917, -0.059856, 0.133934, -0.006038, -0.063217, -0.132800, 0.046318, -0.010909, -0.138195, -0.139630  # dense_weights[659]
    .float 0.027594, -0.057302, -0.060834, -0.107045, 0.102121, -0.021459, 0.067131, 0.036862, 0.007446, 0.054617  # dense_weights[660]
    .float -0.063011, 0.016850, -0.039844, -0.092558, 0.024205, -0.123866, 0.044108, -0.032750, 0.078396, -0.161236  # dense_weights[661]
    .float 0.204762, -0.018768, 0.011891, -0.084081, -0.142778, -0.152370, 0.147437, -0.015103, -0.085091, -0.001449  # dense_weights[662]
    .float -0.023322, -0.059431, -0.066908, -0.045401, 0.069215, -0.004476, -0.004477, -0.055956, -0.046916, 0.019927  # dense_weights[663]
    .float 0.079743, 0.010426, -0.057613, -0.006683, -0.024275, -0.067492, 0.052020, -0.063968, -0.098023, -0.077922  # dense_weights[664]
    .float -0.095329, 0.009160, 0.186961, -0.014652, 0.025479, 0.032594, -0.078963, -0.028260, -0.140450, -0.134791  # dense_weights[665]
    .float -0.084656, -0.056101, 0.180497, 0.069382, 0.009702, -0.001194, -0.104871, 0.009550, -0.018904, -0.203281  # dense_weights[666]
    .float -0.117009, 0.029170, 0.310488, -0.128078, -0.032006, -0.041953, -0.191045, -0.060877, -0.160377, -0.115824  # dense_weights[667]
    .float -0.030653, 0.062385, 0.147761, -0.009671, -0.042459, -0.164705, 0.054661, -0.031412, -0.093615, -0.094844  # dense_weights[668]
    .float 0.031285, -0.043373, 0.286301, -0.155915, -0.000325, -0.189362, -0.012524, -0.010402, -0.094521, -0.275342  # dense_weights[669]
    .float -0.078911, 0.088168, 0.203878, -0.050851, -0.134499, 0.014955, -0.148147, 0.003935, -0.129702, -0.032669  # dense_weights[670]
    .float 0.078873, -0.076954, 0.060344, -0.036738, -0.003238, -0.050025, 0.072215, 0.020638, -0.052228, -0.080190  # dense_weights[671]
    .float 0.005933, 0.025506, -0.051743, 0.080578, -0.005140, 0.148873, -0.057443, -0.031707, -0.117973, -0.005269  # dense_weights[672]
    .float -0.016045, -0.179997, 0.076507, 0.041752, -0.107334, 0.054999, -0.090176, -0.033959, 0.030570, -0.121513  # dense_weights[673]
    .float -0.084609, -0.040817, 0.083754, 0.024370, -0.086206, 0.056271, -0.245922, 0.013592, -0.141775, -0.057604  # dense_weights[674]
    .float 0.081905, -0.048105, 0.169281, -0.019410, 0.054788, -0.079397, -0.149543, 0.023533, -0.049853, -0.094159  # dense_weights[675]
    .float -0.270828, -0.231608, -0.038135, 0.049053, -0.036226, 0.094015, -0.260313, -0.024777, -0.081398, -0.191254  # dense_weights[676]
    .float 0.039610, -0.064816, 0.133185, -0.062819, -0.021559, -0.049345, -0.154125, -0.033034, -0.070888, -0.022761  # dense_weights[677]
    .float 0.077064, -0.018900, 0.171004, 0.047378, 0.029209, -0.095057, -0.035484, -0.047042, -0.083102, -0.154435  # dense_weights[678]
    .float 0.011759, -0.055866, 0.113894, 0.011366, -0.034727, -0.044403, -0.047455, 0.039397, -0.102224, -0.109772  # dense_weights[679]
    .float -0.106433, -0.043927, -0.154640, 0.143950, -0.136991, 0.128783, -0.102969, -0.085107, -0.235831, 0.010168  # dense_weights[680]
    .float 0.026602, -0.071654, 0.081239, 0.049341, -0.008733, 0.016958, -0.051449, -0.022284, 0.046643, -0.089301  # dense_weights[681]
    .float 0.005145, -0.157337, -0.003991, 0.075196, 0.018388, 0.080682, -0.043962, -0.035755, -0.066003, -0.197053  # dense_weights[682]
    .float 0.072094, -0.074566, 0.041092, -0.076599, 0.071493, -0.065940, 0.055676, 0.061382, 0.000497, -0.075452  # dense_weights[683]
    .float -0.163060, -0.136322, -0.014585, 0.067372, -0.011892, 0.187099, -0.182177, -0.146925, -0.134429, -0.026545  # dense_weights[684]
    .float -0.023880, -0.137704, 0.061870, -0.045735, 0.079937, -0.024539, 0.021574, -0.038170, -0.006101, -0.062643  # dense_weights[685]
    .float -0.022867, -0.090481, 0.153088, -0.126720, -0.036598, -0.085675, 0.003117, 0.021263, 0.010671, -0.154359  # dense_weights[686]
    .float 0.009053, -0.098897, 0.032078, 0.036613, 0.113437, -0.081014, 0.001166, 0.015027, 0.052120, -0.026212  # dense_weights[687]
    .float 0.105075, -0.243170, -0.294468, 0.077157, -0.276274, 0.066070, 0.182161, -0.029508, -0.318972, 0.078974  # dense_weights[688]
    .float -0.046281, 0.011111, 0.134354, -0.109519, -0.034708, 0.037433, -0.071308, -0.016731, -0.008633, -0.108885  # dense_weights[689]
    .float -0.021111, -0.117758, 0.053839, 0.072200, 0.037121, 0.128261, -0.138877, 0.002558, -0.019955, -0.080718  # dense_weights[690]
    .float 0.049241, -0.057143, 0.016842, -0.121153, 0.027764, -0.111233, 0.041670, 0.002705, 0.062977, -0.103055  # dense_weights[691]
    .float 0.027021, -0.145921, -0.011305, -0.001228, -0.089434, 0.144853, -0.041216, -0.039431, -0.133714, 0.039756  # dense_weights[692]
    .float 0.069712, -0.044575, -0.005124, -0.089731, 0.083815, -0.002960, 0.026603, -0.028983, -0.101494, -0.048702  # dense_weights[693]
    .float 0.057712, 0.059206, 0.142560, -0.218033, -0.069378, -0.183784, 0.076267, 0.040320, 0.054392, -0.108184  # dense_weights[694]
    .float 0.050109, -0.046503, 0.120421, -0.155064, 0.069009, -0.135385, 0.017330, -0.021724, 0.081764, 0.028835  # dense_weights[695]
    .float 0.122755, -0.168350, -0.264433, 0.136850, -0.071183, 0.040123, 0.195331, 0.070241, -0.321678, 0.073759  # dense_weights[696]
    .float 0.017407, 0.013816, 0.054568, -0.034542, -0.036245, -0.048969, 0.027966, -0.090780, 0.017142, 0.013779  # dense_weights[697]
    .float -0.093052, -0.105401, -0.029119, -0.027464, 0.051060, 0.150470, -0.119190, -0.028190, -0.044872, 0.047564  # dense_weights[698]
    .float 0.069010, -0.035063, 0.065826, -0.201568, -0.152276, -0.124894, 0.018443, 0.063332, 0.139762, -0.020382  # dense_weights[699]
    .float 0.102956, -0.203696, -0.017028, -0.060195, -0.040496, 0.073965, -0.016212, -0.070235, -0.070894, 0.123548  # dense_weights[700]
    .float -0.044238, -0.088441, -0.015419, -0.084893, 0.067193, -0.005692, 0.114755, -0.096518, -0.072415, 0.062477  # dense_weights[701]
    .float 0.110475, -0.053654, 0.138475, -0.202386, -0.180298, -0.271056, 0.098077, 0.197471, 0.139452, -0.267375  # dense_weights[702]
    .float 0.078746, 0.013998, 0.020751, -0.180368, -0.011910, -0.115529, 0.048382, -0.040807, 0.007210, -0.007916  # dense_weights[703]
    .float 0.082608, -0.042715, -0.294692, -0.055800, 0.018628, -0.124503, 0.070962, -0.022085, -0.023782, -0.037137  # dense_weights[704]
    .float -0.045326, -0.100804, -0.002033, 0.042772, -0.047896, 0.046462, -0.071895, -0.117361, -0.053005, 0.099047  # dense_weights[705]
    .float -0.168416, -0.091405, 0.050801, 0.080129, 0.012296, 0.112992, -0.081037, -0.048718, -0.011334, 0.154787  # dense_weights[706]
    .float 0.008792, 0.007087, 0.097569, -0.104520, -0.148766, -0.101999, 0.053933, 0.017789, 0.163247, -0.095456  # dense_weights[707]
    .float 0.030156, -0.280635, 0.009428, -0.086417, -0.060307, 0.087906, 0.052870, -0.191802, 0.045490, 0.059346  # dense_weights[708]
    .float -0.063975, -0.070644, -0.044756, 0.060469, -0.023831, 0.075318, -0.030878, -0.098189, 0.072272, 0.120810  # dense_weights[709]
    .float 0.081323, -0.046373, 0.168612, -0.221974, -0.132981, -0.242113, 0.116626, 0.117302, 0.235087, -0.509940  # dense_weights[710]
    .float 0.070520, -0.111810, 0.085939, -0.169310, 0.024386, -0.014287, 0.112383, -0.161097, -0.019156, -0.024155  # dense_weights[711]
    .float 0.192791, -0.035735, -0.196214, -0.126052, -0.045496, 0.072921, 0.054964, -0.178026, -0.032432, -0.086060  # dense_weights[712]
    .float -0.073235, 0.007651, 0.049872, 0.007614, -0.036167, 0.109831, -0.017658, -0.071594, -0.035695, 0.071403  # dense_weights[713]
    .float -0.148846, 0.030459, 0.036888, 0.118196, -0.035191, 0.057847, -0.043150, -0.074689, -0.069142, 0.129495  # dense_weights[714]
    .float -0.017885, -0.019323, 0.086637, -0.104635, -0.056133, -0.103677, -0.110145, 0.005817, 0.017178, -0.031456  # dense_weights[715]
    .float 0.065537, -0.151226, 0.052692, -0.008953, -0.063506, -0.008188, 0.126224, -0.218025, 0.051155, 0.044155  # dense_weights[716]
    .float -0.226851, 0.076087, 0.063127, 0.022387, 0.057586, -0.002450, -0.040738, -0.130978, 0.066802, 0.068621  # dense_weights[717]
    .float 0.050587, 0.031454, 0.108342, -0.268849, -0.095708, -0.207381, 0.014893, 0.153335, 0.088399, -0.153457  # dense_weights[718]
    .float -0.066454, -0.018427, 0.030940, -0.017814, -0.028425, -0.046365, 0.077250, -0.158028, 0.112619, -0.033909  # dense_weights[719]
    .float -0.112630, 0.014523, 0.043936, -0.014089, -0.187483, 0.081115, -0.004280, -0.183165, 0.040592, -0.137877  # dense_weights[720]
    .float -0.000270, -0.066564, 0.015447, 0.000559, -0.022339, 0.056596, 0.081040, -0.066973, -0.048448, -0.069472  # dense_weights[721]
    .float -0.133337, -0.192937, 0.036142, 0.110769, 0.124848, 0.023384, 0.047192, -0.028971, -0.041875, 0.008943  # dense_weights[722]
    .float 0.019136, -0.091029, 0.031401, -0.181950, -0.023577, 0.017716, -0.065299, -0.008207, -0.190318, 0.066654  # dense_weights[723]
    .float -0.168080, 0.061911, 0.008168, 0.101724, -0.025591, -0.126004, 0.059622, -0.136629, 0.142893, -0.015879  # dense_weights[724]
    .float -0.138619, -0.007781, 0.024366, 0.136670, 0.052951, 0.047386, -0.031539, -0.166457, 0.037021, 0.068868  # dense_weights[725]
    .float 0.107393, 0.081537, 0.112063, -0.208009, -0.000082, -0.052590, 0.023866, 0.108129, -0.196823, -0.042230  # dense_weights[726]
    .float -0.088374, 0.054218, 0.107735, -0.022789, 0.070264, -0.086606, 0.057276, -0.071166, -0.014015, 0.026520  # dense_weights[727]
    .float -0.098305, 0.079120, -0.071019, 0.073972, -0.083934, 0.012952, 0.009473, -0.111651, 0.052871, -0.128909  # dense_weights[728]
    .float 0.086924, -0.097776, 0.005179, 0.114363, -0.064315, 0.010095, 0.015421, 0.069562, -0.040648, -0.113602  # dense_weights[729]
    .float 0.007332, -0.206704, 0.029362, 0.143178, 0.004757, -0.031409, -0.027181, 0.057644, -0.035194, -0.133252  # dense_weights[730]
    .float 0.124073, -0.165189, 0.047038, -0.005378, -0.096416, 0.124628, 0.051914, -0.014695, -0.132492, -0.014430  # dense_weights[731]
    .float -0.089275, 0.098982, 0.037981, -0.004578, 0.033110, -0.028362, -0.009824, -0.042428, 0.050979, 0.032468  # dense_weights[732]
    .float -0.120290, -0.081199, -0.061293, 0.046180, 0.011037, 0.013775, -0.061183, -0.040885, -0.007623, 0.027326  # dense_weights[733]
    .float 0.085339, -0.107557, -0.049213, -0.091933, -0.117174, 0.051691, -0.057677, 0.146954, -0.112420, -0.020388  # dense_weights[734]
    .float -0.001738, -0.054266, 0.016159, -0.103116, 0.066508, 0.005932, 0.035102, 0.036248, -0.027033, 0.024037  # dense_weights[735]
    .float -0.032105, -0.037563, -0.065591, 0.103753, 0.046665, -0.007496, 0.017522, 0.013231, 0.033407, 0.067695  # dense_weights[736]
    .float 0.008671, -0.139036, -0.006768, 0.090847, -0.028581, -0.024228, 0.054387, 0.036940, 0.064764, -0.113463  # dense_weights[737]
    .float 0.032748, -0.097207, 0.073175, 0.065122, 0.059640, -0.007261, 0.029103, 0.134132, 0.029066, -0.101652  # dense_weights[738]
    .float 0.021313, -0.010896, -0.085657, -0.021372, -0.080185, 0.026603, 0.069843, 0.035875, -0.104853, 0.021041  # dense_weights[739]
    .float -0.073035, -0.020072, -0.073407, 0.008518, 0.052482, -0.024202, -0.017009, -0.017883, -0.044049, -0.020540  # dense_weights[740]
    .float -0.018962, -0.254945, 0.028126, 0.075632, -0.004308, 0.046981, 0.057133, 0.006629, -0.011868, -0.065642  # dense_weights[741]
    .float 0.097429, 0.044270, -0.060025, -0.076312, -0.034615, 0.050813, -0.087837, 0.133274, -0.180023, 0.038426  # dense_weights[742]
    .float -0.053324, -0.092943, 0.036579, -0.011257, 0.008650, -0.038679, 0.131553, 0.086295, 0.003264, -0.013920  # dense_weights[743]
    .float -0.003741, -0.027731, -0.160584, 0.000514, 0.002542, 0.062124, 0.062829, 0.062924, -0.039973, -0.023220  # dense_weights[744]
    .float -0.101063, -0.008638, 0.095667, 0.086432, 0.058129, -0.030850, 0.052133, 0.041539, 0.017921, -0.204034  # dense_weights[745]
    .float -0.070133, 0.108920, 0.103349, 0.030834, 0.071887, -0.095925, -0.043384, 0.150569, 0.054835, -0.243211  # dense_weights[746]
    .float 0.043980, 0.012350, -0.007500, -0.017407, -0.037216, -0.013914, 0.067298, 0.068410, -0.129506, -0.010241  # dense_weights[747]
    .float -0.027491, 0.009999, 0.018250, 0.029832, 0.119048, 0.050293, 0.017460, 0.011462, -0.018467, -0.085097  # dense_weights[748]
    .float 0.063155, -0.202532, 0.025137, 0.018343, 0.007542, -0.075318, 0.079142, 0.094467, 0.037830, -0.165642  # dense_weights[749]
    .float 0.161042, 0.064237, -0.164123, -0.053766, -0.066252, 0.028529, 0.003895, 0.031633, -0.121048, 0.140647  # dense_weights[750]
    .float 0.049141, -0.013522, -0.042568, 0.040051, 0.043702, 0.001086, 0.068105, -0.038698, -0.063928, 0.016599  # dense_weights[751]
    .float -0.024158, -0.058713, -0.134271, 0.049618, 0.111930, 0.024005, 0.085549, 0.098820, -0.085841, -0.025016  # dense_weights[752]
    .float 0.010659, 0.108088, 0.063002, -0.035371, 0.058459, -0.047737, -0.084779, -0.108713, -0.016702, -0.189619  # dense_weights[753]
    .float -0.075700, 0.098996, 0.143418, -0.101475, -0.033206, -0.115621, -0.120733, -0.079461, 0.030853, -0.201732  # dense_weights[754]
    .float 0.092579, 0.010672, 0.081062, 0.067862, -0.051569, -0.065117, 0.065448, -0.111338, -0.108345, -0.137812  # dense_weights[755]
    .float -0.068608, 0.002266, -0.053514, -0.077319, -0.062201, 0.003606, 0.008599, 0.064486, -0.017674, -0.068948  # dense_weights[756]
    .float -0.003612, -0.012309, 0.204005, -0.113839, -0.009950, -0.029712, 0.020791, -0.028872, -0.031896, -0.174634  # dense_weights[757]
    .float 0.052847, 0.025099, -0.169071, -0.012459, -0.236826, 0.019642, 0.070078, -0.053065, -0.107156, 0.025659  # dense_weights[758]
    .float 0.005624, 0.024422, 0.027322, 0.003514, 0.009340, -0.066566, 0.018659, -0.036302, -0.001314, 0.008987  # dense_weights[759]
    .float 0.081222, 0.057382, -0.100991, 0.033760, 0.013651, 0.080412, 0.087201, -0.090095, -0.042188, -0.134509  # dense_weights[760]
    .float -0.116091, 0.091355, 0.143190, 0.018328, -0.063978, 0.019303, -0.173241, -0.075997, -0.048212, -0.120199  # dense_weights[761]
    .float -0.116377, -0.006488, 0.186419, -0.091760, -0.093686, -0.096739, -0.138065, -0.041665, 0.096551, -0.186608  # dense_weights[762]
    .float -0.144179, 0.070924, 0.275928, -0.194344, -0.091691, -0.141134, -0.085904, -0.071427, -0.218821, 0.054045  # dense_weights[763]
    .float 0.017691, -0.026384, 0.131211, -0.097547, -0.091352, -0.009006, 0.035780, -0.069801, -0.019646, -0.138105  # dense_weights[764]
    .float 0.060415, 0.097865, 0.297755, -0.078130, -0.122847, -0.155723, -0.044874, 0.017762, -0.132187, -0.286455  # dense_weights[765]
    .float -0.135235, 0.105403, 0.027489, 0.045634, -0.035694, 0.128067, -0.076732, -0.033273, -0.078175, 0.043347  # dense_weights[766]
    .float 0.086063, 0.079452, 0.130210, 0.020212, -0.084883, -0.009100, 0.014462, -0.077135, 0.008974, -0.201989  # dense_weights[767]
    .float -0.037889, 0.091453, 0.045772, -0.058072, -0.142058, 0.189836, -0.051793, 0.071663, -0.057916, -0.034885  # dense_weights[768]
    .float -0.026378, 0.012578, 0.062154, -0.005394, -0.097808, -0.011753, -0.116566, -0.030262, -0.096219, 0.003817  # dense_weights[769]
    .float 0.003884, -0.041184, 0.091352, 0.072588, -0.038844, -0.008920, -0.201315, -0.026795, -0.164289, -0.091863  # dense_weights[770]
    .float -0.063208, -0.130227, 0.072608, 0.051419, -0.074091, -0.049537, -0.165609, -0.027275, -0.090438, -0.029708  # dense_weights[771]
    .float -0.179390, 0.090394, 0.017388, 0.144761, -0.030114, 0.058851, -0.304941, -0.080967, -0.141958, -0.143306  # dense_weights[772]
    .float 0.033528, -0.140549, 0.070828, 0.073571, -0.047188, -0.062611, -0.197457, -0.091398, 0.071393, -0.169829  # dense_weights[773]
    .float -0.030768, -0.004336, 0.164396, -0.003235, -0.079162, -0.002680, -0.171945, -0.026860, -0.020463, -0.088820  # dense_weights[774]
    .float -0.098618, -0.038339, 0.135550, 0.074383, 0.006657, -0.029658, -0.160816, -0.092271, -0.030614, -0.132709  # dense_weights[775]
    .float -0.068053, 0.002796, -0.195348, 0.176571, -0.148812, 0.194928, 0.013349, -0.043638, -0.212681, -0.024981  # dense_weights[776]
    .float -0.058675, -0.031621, 0.002942, 0.022617, -0.055324, 0.098506, -0.067931, -0.015694, -0.006411, -0.020443  # dense_weights[777]
    .float -0.066295, 0.101806, 0.063041, 0.027173, -0.043613, 0.032093, -0.083684, -0.039345, -0.084501, -0.113667  # dense_weights[778]
    .float 0.009909, -0.020630, 0.121853, -0.028577, 0.008235, 0.065612, -0.011289, -0.047258, 0.003746, -0.062360  # dense_weights[779]
    .float -0.151037, 0.111007, 0.031411, 0.109893, -0.095103, 0.089245, -0.254356, -0.092742, -0.163001, 0.017902  # dense_weights[780]
    .float -0.029148, -0.093983, 0.032263, 0.025616, -0.112493, 0.027675, 0.034857, -0.134375, -0.020851, 0.009229  # dense_weights[781]
    .float -0.014825, 0.021368, 0.054875, -0.046942, -0.095450, -0.069363, -0.069513, -0.018358, 0.092466, -0.019915  # dense_weights[782]
    .float -0.001698, -0.002523, 0.014147, 0.047271, 0.074270, -0.074633, -0.096066, -0.106702, 0.070845, -0.011543  # dense_weights[783]
    .float 0.162369, -0.283389, -0.087191, -0.032024, -0.248825, 0.045443, 0.070743, 0.001722, -0.033327, 0.009165  # dense_weights[784]
    .float -0.073885, -0.057404, 0.039804, 0.130320, -0.024786, 0.174081, -0.043042, -0.124056, 0.022499, -0.007005  # dense_weights[785]
    .float -0.015799, 0.021700, -0.047912, 0.110503, 0.053647, 0.089355, -0.102432, 0.015460, -0.134961, 0.026731  # dense_weights[786]
    .float -0.010091, 0.097468, 0.045401, -0.046656, -0.033089, -0.052926, -0.047459, 0.030333, 0.047330, -0.008832  # dense_weights[787]
    .float -0.014858, -0.029056, -0.062750, 0.143814, -0.129649, 0.023711, -0.065644, -0.025879, -0.103152, 0.073308  # dense_weights[788]
    .float 0.051707, -0.013993, 0.046349, 0.028459, 0.034350, -0.016732, -0.001467, -0.013046, 0.007532, -0.008994  # dense_weights[789]
    .float -0.097406, 0.041840, 0.135445, -0.111160, 0.023313, -0.077501, -0.092235, 0.049182, 0.102689, -0.069052  # dense_weights[790]
    .float 0.092674, 0.009223, 0.117446, -0.035342, 0.021202, -0.121329, 0.006697, -0.009230, 0.099547, -0.034697  # dense_weights[791]
    .float 0.112829, -0.194355, -0.046758, -0.029652, -0.141647, -0.064184, 0.174425, 0.079373, 0.113255, -0.061811  # dense_weights[792]
    .float -0.038531, 0.044701, -0.010122, 0.083882, -0.058342, 0.126821, 0.047211, -0.114699, -0.115224, 0.073917  # dense_weights[793]
    .float -0.072861, 0.063451, -0.027037, 0.104779, -0.030094, 0.226378, -0.058081, -0.042162, -0.104184, 0.028242  # dense_weights[794]
    .float -0.082774, -0.036907, 0.055228, -0.063157, -0.101673, -0.005303, 0.054741, -0.089248, 0.042935, -0.072848  # dense_weights[795]
    .float 0.104062, -0.065912, -0.021244, 0.023452, -0.093322, 0.025533, -0.006650, -0.116683, -0.042078, 0.047341  # dense_weights[796]
    .float 0.059390, 0.040557, 0.071615, 0.021990, -0.083082, -0.025873, 0.038000, -0.165835, -0.095374, 0.066983  # dense_weights[797]
    .float 0.035604, -0.010312, 0.175027, -0.189169, -0.095484, -0.183865, -0.023684, 0.008768, 0.068149, -0.116426  # dense_weights[798]
    .float 0.020426, -0.033411, 0.050123, -0.032645, -0.087125, -0.076645, 0.049076, -0.083249, 0.042972, 0.006652  # dense_weights[799]
    .float 0.113704, -0.010726, 0.013729, -0.021352, -0.106163, -0.080771, 0.224422, -0.017221, 0.131485, -0.175094  # dense_weights[800]
    .float 0.016419, 0.044296, -0.000680, -0.013597, -0.117870, 0.192775, 0.051421, -0.058887, -0.057696, 0.030697  # dense_weights[801]
    .float -0.056705, 0.104394, -0.031239, 0.113927, -0.040084, 0.133942, -0.114514, -0.091207, -0.026728, 0.053996  # dense_weights[802]
    .float -0.071207, -0.034635, 0.014400, -0.135739, -0.036717, -0.021689, 0.077032, 0.008019, 0.055420, -0.131241  # dense_weights[803]
    .float 0.116240, -0.179031, 0.088382, 0.017071, -0.211225, 0.040838, 0.130313, -0.183051, 0.085443, -0.011490  # dense_weights[804]
    .float 0.040852, -0.016299, -0.002508, -0.069433, 0.014765, 0.065773, 0.057793, -0.070585, 0.000037, 0.076596  # dense_weights[805]
    .float 0.095045, -0.000188, 0.213407, -0.223606, -0.039666, -0.261376, 0.111145, 0.061977, 0.174915, -0.209174  # dense_weights[806]
    .float 0.056863, -0.068860, 0.037067, -0.089437, -0.008056, 0.010819, 0.074232, -0.096307, 0.030159, -0.010300  # dense_weights[807]
    .float 0.074513, 0.116755, 0.007253, -0.186598, -0.082967, -0.114146, 0.170363, -0.033088, 0.048740, -0.228037  # dense_weights[808]
    .float 0.029914, -0.056149, 0.039911, 0.073400, 0.014812, 0.052932, 0.030959, -0.006812, -0.175545, -0.029468  # dense_weights[809]
    .float 0.000919, -0.022499, 0.068746, 0.129215, -0.112787, 0.170682, 0.016013, -0.098529, -0.140931, 0.003832  # dense_weights[810]
    .float -0.063988, 0.100605, -0.000451, 0.025607, -0.013203, 0.019253, -0.064740, -0.012720, 0.047563, 0.029735  # dense_weights[811]
    .float -0.025260, -0.061510, 0.086815, 0.005938, -0.170369, -0.089751, 0.116434, -0.220406, 0.158863, -0.127884  # dense_weights[812]
    .float -0.014446, 0.053118, 0.012552, 0.080352, 0.049526, -0.062677, 0.014801, -0.030931, -0.078701, 0.071820  # dense_weights[813]
    .float 0.023614, 0.218427, 0.069233, -0.225265, 0.055364, -0.078997, -0.058396, 0.163702, 0.026627, -0.099711  # dense_weights[814]
    .float -0.059992, 0.029700, 0.102809, -0.037673, -0.091245, -0.062946, 0.056574, -0.100834, 0.039520, -0.052854  # dense_weights[815]
    .float -0.169357, 0.075215, -0.057392, -0.010721, -0.123758, -0.007107, -0.036306, 0.016459, 0.048930, -0.099597  # dense_weights[816]
    .float 0.022958, -0.112159, 0.066675, 0.171779, -0.109208, 0.060351, 0.114103, 0.004482, -0.036865, -0.043868  # dense_weights[817]
    .float -0.035471, -0.126461, 0.054393, 0.131994, -0.045864, 0.121507, 0.031193, -0.087051, -0.146088, -0.087318  # dense_weights[818]
    .float 0.010469, -0.103279, -0.013155, 0.076901, -0.090893, 0.113307, -0.018237, -0.083073, -0.051660, -0.072357  # dense_weights[819]
    .float -0.154412, 0.085916, 0.093022, 0.024130, -0.050360, -0.050767, 0.067903, -0.046378, 0.082210, -0.081405  # dense_weights[820]
    .float -0.071589, 0.033977, 0.105769, 0.004592, 0.053847, 0.018752, 0.055791, -0.071035, 0.003806, 0.064746  # dense_weights[821]
    .float 0.016334, 0.065815, 0.081326, -0.187548, -0.014464, 0.055103, -0.174993, 0.106510, -0.136201, -0.049673  # dense_weights[822]
    .float -0.093772, 0.041132, 0.060585, -0.012537, 0.064564, 0.006724, 0.011011, 0.053753, -0.025956, -0.028654  # dense_weights[823]
    .float -0.051951, 0.053603, -0.082154, -0.021111, -0.047711, 0.028784, -0.033278, -0.054701, 0.071718, -0.059479  # dense_weights[824]
    .float 0.073798, 0.071719, 0.031251, 0.064708, -0.124559, 0.060208, -0.010514, -0.099107, 0.040891, -0.100548  # dense_weights[825]
    .float 0.009952, -0.105505, 0.027546, 0.072027, -0.077391, 0.009728, -0.058895, -0.131833, -0.066024, -0.127065  # dense_weights[826]
    .float 0.031870, -0.108176, 0.018133, 0.109157, -0.121656, 0.101861, 0.069599, -0.091255, -0.050172, 0.054293  # dense_weights[827]
    .float -0.033249, 0.072577, 0.033355, 0.005610, -0.015094, -0.078382, 0.053839, -0.046027, 0.024269, -0.025768  # dense_weights[828]
    .float 0.028866, -0.217091, 0.006266, 0.074543, 0.006004, 0.062015, 0.006842, -0.018928, -0.008839, -0.026659  # dense_weights[829]
    .float 0.029070, -0.208787, -0.031081, -0.007894, -0.135234, 0.115799, -0.077313, 0.118186, -0.042774, 0.036685  # dense_weights[830]
    .float -0.041269, -0.093386, 0.015101, 0.029450, 0.096874, 0.035800, 0.067556, -0.032233, -0.013946, 0.057878  # dense_weights[831]
    .float -0.056736, -0.020034, -0.060435, 0.049807, -0.053857, 0.050049, 0.065646, -0.045627, 0.074033, -0.064381  # dense_weights[832]
    .float -0.003954, 0.025227, 0.028279, 0.016390, -0.031609, -0.057580, -0.043814, -0.029396, -0.006754, -0.112483  # dense_weights[833]
    .float -0.038532, 0.110026, 0.252101, 0.033413, -0.059781, 0.055213, -0.063112, -0.080901, -0.080799, -0.228822  # dense_weights[834]
    .float 0.081672, -0.040479, -0.053352, 0.020689, -0.119075, 0.075149, -0.025605, -0.060672, 0.061169, 0.074631  # dense_weights[835]
    .float -0.001786, -0.087777, 0.102376, 0.084700, 0.000248, -0.086235, 0.047347, 0.005459, 0.007946, -0.053985  # dense_weights[836]
    .float 0.014547, -0.120268, 0.110211, 0.112324, -0.031631, 0.040580, 0.035340, -0.088671, -0.040686, -0.155138  # dense_weights[837]
    .float 0.047284, -0.212917, -0.245514, 0.044393, -0.114537, 0.047068, -0.049026, 0.123735, -0.094434, -0.043273  # dense_weights[838]
    .float -0.032347, -0.008209, -0.059558, -0.024495, -0.029820, 0.064530, 0.049534, -0.002061, -0.090434, -0.023432  # dense_weights[839]
    .float -0.064708, 0.134034, 0.027594, 0.033156, 0.024220, 0.064959, 0.056449, -0.026459, 0.023637, -0.059221  # dense_weights[840]
    .float 0.001616, 0.129880, 0.174038, -0.058029, -0.073268, -0.091979, -0.051508, 0.006066, 0.041036, -0.074054  # dense_weights[841]
    .float -0.000411, 0.190624, 0.166588, -0.097602, -0.117075, 0.019051, 0.004357, -0.039507, -0.003148, -0.259475  # dense_weights[842]
    .float 0.030461, -0.001973, -0.081839, 0.121137, -0.134002, 0.090864, 0.062809, -0.027428, 0.041180, -0.112516  # dense_weights[843]
    .float -0.002384, 0.044554, 0.119168, -0.056890, -0.000203, -0.040764, 0.124537, 0.000556, 0.034144, -0.115770  # dense_weights[844]
    .float -0.042764, -0.045547, 0.145366, -0.053393, -0.058462, 0.000355, 0.059911, -0.082500, -0.033792, -0.111139  # dense_weights[845]
    .float 0.119996, -0.239329, -0.213870, 0.112168, -0.260804, 0.056092, -0.032316, -0.019163, -0.011888, -0.044804  # dense_weights[846]
    .float -0.008532, 0.052248, 0.063299, 0.025496, 0.032909, -0.036600, 0.026119, -0.060917, -0.090568, -0.067783  # dense_weights[847]
    .float -0.065566, 0.082820, 0.011661, -0.021043, -0.018284, 0.001077, 0.056506, -0.058653, -0.019663, -0.014349  # dense_weights[848]
    .float -0.103246, 0.034262, 0.095930, -0.077760, -0.096225, 0.025046, -0.152791, -0.177896, -0.049591, -0.128410  # dense_weights[849]
    .float -0.039351, 0.113266, 0.184755, -0.117968, -0.096355, -0.003272, 0.004000, -0.108781, -0.119774, -0.071082  # dense_weights[850]
    .float -0.024363, 0.026229, 0.028334, 0.039390, -0.220180, -0.014823, 0.034195, -0.204741, -0.000192, -0.133573  # dense_weights[851]
    .float 0.043193, 0.034404, 0.029979, 0.027487, -0.052411, -0.010133, -0.025170, -0.009235, -0.073136, -0.058684  # dense_weights[852]
    .float 0.041479, 0.082238, 0.193687, -0.041597, -0.007326, -0.086825, 0.019732, -0.059817, -0.028685, -0.051459  # dense_weights[853]
    .float 0.015592, 0.100785, -0.185188, 0.086145, -0.163525, 0.187602, 0.050869, -0.048372, -0.027165, 0.009406  # dense_weights[854]
    .float -0.050223, 0.044127, 0.036928, 0.066948, -0.056966, 0.059161, 0.076005, -0.007250, -0.006266, -0.066710  # dense_weights[855]
    .float 0.038359, -0.014664, 0.007768, 0.002266, -0.107535, 0.027952, 0.056595, -0.065030, -0.035535, -0.049174  # dense_weights[856]
    .float -0.107477, 0.018074, 0.229603, -0.177720, -0.140479, -0.020096, -0.087000, -0.160715, -0.030243, -0.085069  # dense_weights[857]
    .float -0.224790, 0.097707, 0.233744, -0.150221, -0.093333, -0.108117, -0.116694, -0.176010, -0.088651, -0.028693  # dense_weights[858]
    .float -0.109710, 0.079909, 0.200374, -0.054095, -0.081432, 0.038593, -0.120108, 0.040676, -0.256361, 0.011736  # dense_weights[859]
    .float 0.073955, 0.047098, 0.046219, 0.003576, 0.020446, 0.017620, -0.024082, -0.154014, -0.068595, -0.126786  # dense_weights[860]
    .float -0.093965, 0.018356, 0.174284, -0.024551, -0.132798, -0.116150, -0.104832, -0.119536, 0.038517, -0.212659  # dense_weights[861]
    .float 0.005394, 0.053492, -0.062184, 0.028744, -0.141698, 0.101367, -0.020118, 0.063738, -0.201608, 0.077970  # dense_weights[862]
    .float 0.036507, 0.024913, 0.114574, -0.072990, -0.091899, 0.047352, -0.011849, -0.158274, 0.092119, -0.020473  # dense_weights[863]
    .float -0.047668, 0.039833, 0.116644, 0.003350, -0.081947, 0.171058, -0.100660, 0.053683, -0.165775, 0.038895  # dense_weights[864]
    .float 0.054089, 0.161195, 0.097807, 0.086377, -0.175755, 0.005819, -0.134845, 0.048494, -0.118173, -0.073414  # dense_weights[865]
    .float -0.104643, 0.081381, 0.016265, 0.022138, -0.150451, -0.009357, -0.149595, -0.044675, -0.135386, -0.076851  # dense_weights[866]
    .float -0.070146, 0.095530, 0.107118, -0.047767, -0.081544, -0.047772, -0.088285, -0.013844, 0.012555, -0.023776  # dense_weights[867]
    .float -0.282511, 0.142066, -0.050162, 0.065627, -0.071318, 0.094841, -0.253387, 0.109139, -0.199996, -0.114560  # dense_weights[868]
    .float -0.079753, 0.087535, 0.064997, 0.054441, -0.085795, -0.138823, -0.162067, 0.042157, -0.077048, -0.149929  # dense_weights[869]
    .float -0.049131, -0.020921, 0.021702, 0.030647, -0.039253, -0.017399, -0.128820, 0.046668, 0.023477, 0.016315  # dense_weights[870]
    .float -0.018084, 0.028200, 0.121490, 0.047622, -0.119677, 0.019250, -0.153237, -0.023706, -0.125661, -0.129570  # dense_weights[871]
    .float 0.011311, -0.083361, 0.002620, 0.100839, -0.067596, 0.079902, -0.009023, -0.122400, -0.211627, -0.025344  # dense_weights[872]
    .float 0.035076, 0.082203, -0.008359, -0.004128, -0.010165, 0.009151, -0.130288, -0.010587, 0.034535, -0.035033  # dense_weights[873]
    .float -0.043379, 0.051404, 0.025346, -0.019293, -0.120973, 0.125338, -0.187236, -0.037407, -0.062482, -0.095127  # dense_weights[874]
    .float 0.010981, 0.147794, 0.044460, -0.027758, -0.050723, -0.084554, -0.065289, 0.055802, 0.030556, -0.020347  # dense_weights[875]
    .float -0.020823, 0.058310, -0.119745, 0.076766, 0.043619, 0.090270, -0.304174, 0.076034, -0.062537, -0.019748  # dense_weights[876]
    .float 0.068892, 0.026290, -0.003863, 0.104886, -0.062425, 0.035689, -0.011211, -0.056075, 0.084518, 0.020650  # dense_weights[877]
    .float -0.016296, 0.030021, 0.030231, 0.016140, -0.011530, 0.052965, -0.043291, 0.037679, -0.022081, 0.019854  # dense_weights[878]
    .float 0.069282, -0.033325, 0.018475, -0.033196, -0.027878, 0.015511, -0.021587, -0.038866, -0.029142, -0.129011  # dense_weights[879]
    .float 0.108530, -0.201920, 0.043061, -0.157646, -0.148446, -0.015393, -0.157966, -0.035712, 0.257511, -0.096680  # dense_weights[880]
    .float -0.046918, 0.115128, 0.002866, 0.025130, -0.078137, 0.057702, -0.079146, -0.052742, 0.017215, -0.031895  # dense_weights[881]
    .float -0.072404, 0.145353, -0.025820, 0.045269, 0.042756, 0.105705, -0.137657, 0.001975, -0.102073, 0.061211  # dense_weights[882]
    .float -0.023115, 0.061593, 0.061175, -0.039771, 0.038958, 0.087483, -0.035195, 0.073657, -0.022505, -0.001656  # dense_weights[883]
    .float -0.006954, -0.080546, -0.063604, 0.071632, 0.009502, 0.064266, 0.004633, -0.064927, -0.012787, 0.156416  # dense_weights[884]
    .float 0.057922, -0.015151, 0.079829, 0.065272, -0.112589, 0.074069, -0.023839, -0.145362, -0.083155, 0.062419  # dense_weights[885]
    .float -0.034373, 0.089644, 0.115173, -0.004127, -0.045674, 0.033230, -0.104696, 0.095892, -0.045156, -0.074276  # dense_weights[886]
    .float 0.077111, -0.044901, 0.114024, 0.014477, -0.028345, -0.022780, -0.057576, 0.005672, 0.065343, -0.006373  # dense_weights[887]
    .float 0.232886, -0.070096, 0.036643, -0.185981, -0.062718, -0.210172, 0.149580, -0.206424, 0.263281, -0.085870  # dense_weights[888]
    .float 0.045776, -0.033147, -0.022539, 0.097993, -0.032154, 0.094510, -0.036954, -0.040035, -0.047533, 0.069737  # dense_weights[889]
    .float -0.112930, 0.105834, -0.038258, 0.040118, -0.087175, 0.124795, -0.106169, -0.053129, -0.106891, 0.114302  # dense_weights[890]
    .float 0.023643, -0.020346, -0.024289, -0.052468, -0.037401, 0.038200, -0.013849, 0.008640, -0.014287, 0.029967  # dense_weights[891]
    .float 0.007268, -0.113213, 0.082347, 0.082114, -0.090596, -0.017969, 0.093911, -0.177878, 0.026808, -0.028654  # dense_weights[892]
    .float 0.094740, -0.022808, -0.043515, -0.010813, -0.092221, 0.133432, 0.066042, -0.215599, -0.055396, -0.080199  # dense_weights[893]
    .float 0.074694, -0.056516, 0.024399, -0.151294, 0.041701, -0.010641, -0.052315, 0.039823, -0.053956, -0.002556  # dense_weights[894]
    .float 0.047440, 0.000351, 0.123039, 0.008317, 0.021508, -0.047632, 0.080055, -0.070168, -0.028637, -0.085009  # dense_weights[895]
    .float 0.147953, -0.046791, 0.065717, -0.268006, -0.061448, -0.136355, 0.167505, 0.047261, 0.195547, -0.048275  # dense_weights[896]
    .float -0.035952, -0.022434, -0.064703, 0.062161, -0.112965, 0.167177, 0.012578, -0.042303, -0.011740, 0.003750  # dense_weights[897]
    .float -0.013734, 0.051814, -0.029839, 0.126021, -0.011962, 0.169295, -0.042595, -0.071839, -0.101436, -0.005067  # dense_weights[898]
    .float -0.044444, -0.050739, -0.020426, 0.018077, 0.037519, 0.050721, 0.045557, -0.042304, -0.065683, -0.012091  # dense_weights[899]
    .float 0.103455, -0.205595, 0.126426, 0.025598, -0.164074, -0.030952, 0.068226, -0.254433, 0.166280, -0.050718  # dense_weights[900]
    .float 0.100046, 0.009522, -0.055607, -0.013602, -0.107878, 0.106348, 0.065793, -0.173373, -0.062286, 0.026531  # dense_weights[901]
    .float 0.028897, 0.065627, 0.081993, -0.245791, 0.072271, -0.045913, -0.117566, 0.034900, 0.038378, -0.005214  # dense_weights[902]
    .float 0.053272, 0.014400, 0.099704, 0.038089, -0.064396, 0.050885, 0.049985, -0.115199, 0.050790, -0.040508  # dense_weights[903]
    .float 0.012894, 0.161379, 0.047912, -0.315671, 0.019124, -0.097286, 0.033511, 0.067370, 0.132215, -0.037282  # dense_weights[904]
    .float -0.011758, -0.108929, -0.040513, 0.076000, -0.076275, 0.082298, 0.085229, -0.113778, -0.066843, 0.016571  # dense_weights[905]
    .float 0.056321, -0.052295, -0.063495, 0.049762, -0.084872, 0.093348, -0.085767, -0.068998, -0.014026, 0.089054  # dense_weights[906]
    .float -0.084713, -0.144456, -0.038356, 0.011896, 0.006582, -0.006106, 0.019234, 0.027955, 0.097507, -0.033808  # dense_weights[907]
    .float 0.079575, -0.048888, 0.094742, -0.089377, -0.113328, 0.022127, 0.094715, -0.126414, 0.040175, 0.002508  # dense_weights[908]
    .float 0.083431, -0.057638, 0.052393, -0.003745, -0.050086, 0.005342, 0.177456, -0.146409, -0.041019, -0.067540  # dense_weights[909]
    .float -0.272029, 0.104335, -0.067282, -0.158903, 0.027186, 0.025098, -0.274143, 0.091851, -0.033426, 0.120104  # dense_weights[910]
    .float -0.051767, -0.034135, -0.002072, 0.028394, -0.096508, 0.017082, 0.123846, -0.043584, -0.079446, -0.039966  # dense_weights[911]
    .float -0.093167, 0.014026, -0.041471, -0.142734, 0.022582, -0.086135, 0.005263, 0.100020, 0.038488, -0.047825  # dense_weights[912]
    .float 0.014382, -0.098495, -0.020127, 0.087351, -0.188539, 0.058427, 0.041399, -0.196821, 0.004005, 0.011905  # dense_weights[913]
    .float 0.035071, -0.121616, 0.029385, 0.061907, -0.164622, 0.037816, -0.051531, -0.131649, -0.009714, 0.107296  # dense_weights[914]
    .float -0.063941, -0.015441, -0.102385, 0.073853, -0.012065, 0.016987, -0.150679, -0.110831, 0.051222, -0.006154  # dense_weights[915]
    .float -0.013165, 0.104747, 0.105841, -0.019732, -0.109591, -0.018956, 0.039943, 0.036450, 0.005088, -0.066847  # dense_weights[916]
    .float -0.060830, -0.036956, -0.015725, -0.015680, -0.065105, 0.061752, 0.123148, -0.023796, 0.015536, -0.034758  # dense_weights[917]
    .float -0.185238, 0.001861, -0.147090, -0.002128, 0.043232, 0.043661, -0.273441, 0.160974, -0.092615, 0.107771  # dense_weights[918]
    .float -0.048274, -0.043489, 0.008716, -0.022573, -0.093509, 0.061331, 0.068974, -0.038339, 0.013759, 0.018174  # dense_weights[919]
    .float -0.048292, -0.011132, -0.042951, -0.065974, 0.030567, -0.071052, -0.073047, -0.012772, 0.032283, -0.003217  # dense_weights[920]
    .float 0.052039, 0.026614, 0.091071, 0.023856, -0.134966, -0.021939, 0.056841, -0.146593, 0.064470, -0.103763  # dense_weights[921]
    .float 0.063793, -0.096011, 0.068031, 0.027083, -0.254410, 0.120853, 0.057403, -0.160108, -0.061883, -0.036971  # dense_weights[922]
    .float -0.022915, -0.218872, -0.091155, 0.023718, -0.075451, 0.100763, -0.100509, -0.126739, 0.063290, -0.048045  # dense_weights[923]
    .float 0.036309, 0.047295, -0.000137, 0.038803, -0.005991, -0.063807, 0.121380, -0.080653, -0.052112, -0.073467  # dense_weights[924]
    .float -0.024853, 0.000744, 0.173220, 0.085302, -0.083032, -0.063453, 0.146380, -0.180093, -0.015389, -0.175879  # dense_weights[925]
    .float -0.098281, -0.087106, -0.322632, -0.012769, 0.029471, 0.129220, -0.157706, 0.018041, -0.035463, 0.005420  # dense_weights[926]
    .float 0.012842, 0.004699, -0.032540, 0.077281, 0.000595, -0.057025, 0.082192, -0.000458, 0.073578, -0.108326  # dense_weights[927]
    .float -0.048287, 0.047187, 0.044955, 0.056658, -0.002515, -0.016989, 0.054885, -0.090060, -0.028479, -0.018632  # dense_weights[928]
    .float -0.017166, 0.115428, 0.167172, 0.023717, -0.089680, -0.029341, -0.061640, -0.142817, -0.053175, -0.117023  # dense_weights[929]
    .float -0.000785, -0.022621, 0.177524, 0.036417, -0.109480, -0.021969, -0.021294, -0.120421, -0.082741, -0.108551  # dense_weights[930]
    .float -0.001945, -0.140215, -0.050028, 0.072734, -0.074604, -0.001008, -0.051059, -0.040933, 0.025894, 0.026532  # dense_weights[931]
    .float -0.053521, 0.009002, 0.047529, 0.035682, 0.062571, -0.024314, 0.147471, -0.039267, 0.055588, -0.113975  # dense_weights[932]
    .float -0.099378, 0.063299, 0.085549, -0.024893, -0.099766, -0.034527, 0.041887, -0.237409, -0.007870, -0.164220  # dense_weights[933]
    .float -0.080459, -0.142159, -0.378062, 0.052192, -0.018594, 0.060171, -0.136075, 0.047703, 0.022557, -0.041841  # dense_weights[934]
    .float -0.028184, 0.126455, 0.067988, 0.085812, 0.001676, 0.031994, 0.083750, -0.050005, 0.014258, -0.089967  # dense_weights[935]
    .float -0.053497, 0.116589, 0.181941, 0.047487, 0.010510, -0.043086, 0.055572, -0.141465, -0.008404, -0.089622  # dense_weights[936]
    .float -0.068902, 0.070796, 0.210392, -0.074824, 0.019218, -0.013967, -0.030473, -0.135127, -0.032002, -0.051183  # dense_weights[937]
    .float 0.017299, 0.015610, 0.268921, 0.022339, -0.095344, 0.058300, -0.007068, -0.135632, -0.048741, -0.209837  # dense_weights[938]
    .float 0.092312, -0.207178, -0.038393, 0.081277, -0.102761, 0.132271, -0.039611, -0.172255, -0.027989, -0.025046  # dense_weights[939]
    .float 0.023171, 0.075607, 0.002690, 0.013368, -0.036646, -0.062209, 0.112389, -0.120748, 0.041217, -0.048517  # dense_weights[940]
    .float -0.008761, 0.082313, 0.116862, -0.025959, -0.089161, 0.011832, -0.028910, -0.153008, -0.061680, -0.070895  # dense_weights[941]
    .float -0.006486, -0.050824, -0.385600, 0.031985, -0.053649, 0.081265, -0.159512, 0.044925, 0.108273, 0.029208  # dense_weights[942]
    .float -0.036013, 0.111335, 0.082236, -0.061498, 0.048544, 0.055274, 0.086937, -0.121564, 0.051752, -0.068949  # dense_weights[943]
    .float -0.083711, 0.007573, -0.031352, -0.021719, 0.057275, 0.040256, -0.018733, -0.033588, 0.030123, 0.041761  # dense_weights[944]
    .float -0.014023, 0.037088, 0.144593, -0.149285, -0.008407, -0.067693, -0.124458, -0.255759, -0.063094, -0.007767  # dense_weights[945]
    .float -0.033591, 0.027537, 0.224117, -0.225788, 0.028527, -0.066955, -0.059406, -0.212657, -0.162963, -0.037153  # dense_weights[946]
    .float -0.136410, -0.062680, 0.039375, 0.049515, 0.015933, 0.110867, -0.136025, -0.039287, -0.057505, -0.065012  # dense_weights[947]
    .float 0.030625, 0.082912, 0.030510, -0.068640, 0.032353, 0.014768, -0.059753, -0.190764, -0.024222, 0.065575  # dense_weights[948]
    .float -0.006085, 0.124208, 0.123359, -0.092847, -0.069841, 0.037709, -0.107607, -0.158215, 0.024717, -0.121331  # dense_weights[949]
    .float -0.170924, 0.090360, -0.223676, -0.006740, -0.069121, 0.282070, -0.130983, 0.023168, -0.122718, 0.077829  # dense_weights[950]
    .float -0.025120, 0.109921, -0.036842, -0.063078, 0.044488, 0.000429, -0.061936, -0.098462, 0.043294, 0.040234  # dense_weights[951]
    .float -0.005690, 0.016887, -0.087216, -0.051223, -0.093023, 0.009545, 0.056099, -0.169571, 0.057219, 0.036774  # dense_weights[952]
    .float -0.051760, -0.066471, 0.184870, -0.117688, -0.141877, 0.028347, -0.104240, -0.121223, -0.062670, 0.041065  # dense_weights[953]
    .float -0.089115, -0.060171, 0.287201, -0.273670, 0.020314, -0.082770, -0.205322, -0.143499, -0.241670, 0.118194  # dense_weights[954]
    .float -0.120935, 0.003828, 0.017657, 0.022269, -0.108322, 0.179276, -0.086699, -0.041384, -0.219292, 0.044558  # dense_weights[955]
    .float -0.015385, -0.014883, 0.025846, -0.087639, -0.017859, 0.009843, -0.032687, -0.157609, 0.086660, 0.006624  # dense_weights[956]
    .float -0.058072, 0.046590, 0.061019, -0.100831, -0.120699, -0.024626, -0.085289, -0.156902, 0.032176, 0.030354  # dense_weights[957]
    .float -0.008426, 0.007903, -0.072233, 0.016684, -0.042710, 0.171233, -0.038687, 0.115931, -0.178926, -0.014596  # dense_weights[958]
    .float -0.052892, -0.052388, 0.005788, -0.047189, -0.056395, 0.076117, -0.021798, -0.172402, 0.084982, 0.027152  # dense_weights[959]
    .float -0.020345, -0.029447, 0.006621, 0.037328, -0.056388, 0.138650, -0.064064, -0.016230, -0.077837, -0.070698  # dense_weights[960]
    .float -0.024128, -0.042395, -0.025402, 0.183274, -0.106726, 0.034483, -0.082920, 0.077111, -0.224620, 0.036768  # dense_weights[961]
    .float -0.106824, -0.051684, -0.019631, 0.205670, -0.190756, 0.109468, -0.177956, 0.036870, -0.227820, 0.005232  # dense_weights[962]
    .float 0.020610, 0.125782, -0.009487, 0.050565, -0.134762, 0.024869, -0.195291, 0.048447, -0.036656, -0.001724  # dense_weights[963]
    .float -0.094187, -0.160272, -0.283978, 0.189721, 0.020494, 0.055455, -0.034542, 0.129662, -0.323212, -0.082148  # dense_weights[964]
    .float -0.071650, 0.188100, 0.078152, -0.024051, -0.005788, -0.057911, -0.262126, 0.000997, -0.047365, 0.006827  # dense_weights[965]
    .float 0.035051, 0.108522, 0.044413, 0.057880, -0.156774, -0.041459, -0.191048, 0.061647, -0.014162, 0.036891  # dense_weights[966]
    .float -0.097664, 0.036136, 0.107329, 0.039321, -0.000729, -0.015568, -0.102827, -0.133937, -0.068869, 0.019235  # dense_weights[967]
    .float -0.017215, 0.047979, 0.087723, 0.001725, -0.108526, 0.129093, -0.010293, -0.104403, -0.095439, -0.030097  # dense_weights[968]
    .float -0.145004, 0.070500, -0.095403, 0.081302, -0.014031, 0.018910, -0.137323, 0.112765, -0.090360, 0.043446  # dense_weights[969]
    .float -0.113384, 0.003932, -0.104432, 0.110170, -0.020670, 0.090967, -0.077167, 0.032070, -0.143757, -0.031654  # dense_weights[970]
    .float -0.045169, 0.104862, 0.019689, -0.041238, -0.003401, -0.060095, -0.200483, 0.188356, -0.082499, 0.057833  # dense_weights[971]
    .float -0.050072, -0.109548, -0.117893, 0.149007, -0.005613, 0.009760, -0.190234, 0.162395, -0.241623, -0.017584  # dense_weights[972]
    .float 0.010373, 0.051931, -0.028491, 0.025049, -0.050145, -0.149914, -0.138961, 0.002257, 0.014500, -0.035088  # dense_weights[973]
    .float 0.000273, 0.112750, -0.052765, 0.038560, -0.127207, -0.026442, -0.163608, 0.175379, 0.022676, 0.027477  # dense_weights[974]
    .float 0.021217, -0.005593, 0.069918, 0.037465, -0.035235, -0.035841, -0.176341, 0.043224, 0.033804, 0.039331  # dense_weights[975]
    .float 0.081466, 0.029779, 0.006911, -0.015956, -0.092774, 0.115433, -0.143831, 0.016280, 0.000888, -0.174446  # dense_weights[976]
    .float -0.062598, 0.073427, -0.137804, 0.161451, 0.024018, 0.145712, -0.197717, 0.110564, -0.083773, 0.025942  # dense_weights[977]
    .float -0.065606, -0.030797, -0.012657, 0.262321, -0.023772, 0.197826, -0.047984, 0.006296, -0.106863, 0.070335  # dense_weights[978]
    .float -0.023476, 0.020189, -0.065165, 0.020854, -0.015733, 0.001299, -0.041637, 0.070383, -0.000310, 0.042409  # dense_weights[979]
    .float 0.046797, -0.093252, 0.073498, 0.078302, -0.054229, 0.051831, -0.174031, -0.066415, 0.077808, 0.035425  # dense_weights[980]
    .float 0.033471, 0.111357, -0.027609, 0.004015, -0.067194, -0.079685, -0.113430, -0.083793, 0.056350, -0.015996  # dense_weights[981]
    .float -0.093833, -0.011708, 0.068209, 0.027012, -0.019021, 0.064292, -0.116916, 0.148804, -0.086661, -0.003908  # dense_weights[982]
    .float 0.024115, 0.047800, 0.065662, 0.087722, 0.034524, 0.020256, -0.083566, 0.041624, 0.036500, -0.016937  # dense_weights[983]
    .float 0.053388, 0.103889, 0.012023, -0.125425, 0.215075, -0.089138, -0.256103, 0.062102, 0.202836, -0.247175  # dense_weights[984]
    .float -0.072141, -0.035829, -0.129177, 0.017960, 0.005786, 0.123214, -0.301533, -0.003042, -0.125096, 0.064785  # dense_weights[985]
    .float -0.082497, -0.111280, -0.134427, 0.169811, -0.115100, 0.115497, 0.002283, -0.050856, -0.086153, 0.084217  # dense_weights[986]
    .float -0.071085, -0.066243, -0.040587, 0.063227, 0.032699, 0.132494, -0.059759, -0.016313, -0.077270, 0.014738  # dense_weights[987]
    .float -0.019203, -0.154098, 0.024925, 0.080051, -0.082726, 0.071208, -0.045378, -0.152154, 0.072440, -0.118911  # dense_weights[988]
    .float 0.006159, 0.067695, 0.072269, 0.073969, 0.037924, -0.016702, -0.092698, -0.145468, 0.015291, -0.032801  # dense_weights[989]
    .float -0.066965, 0.080298, -0.037008, -0.071101, 0.040635, 0.005521, -0.084373, 0.056777, -0.195438, 0.020653  # dense_weights[990]
    .float 0.039012, 0.026344, 0.052190, 0.079481, -0.100118, -0.002620, -0.066441, -0.057651, 0.028903, -0.010768  # dense_weights[991]
    .float 0.032671, 0.299362, -0.143243, -0.278202, 0.188087, -0.026312, -0.077183, 0.193916, 0.115058, -0.080003  # dense_weights[992]
    .float -0.063124, -0.134766, -0.021766, 0.091655, -0.057614, 0.172142, -0.221619, -0.039264, 0.022762, -0.031843  # dense_weights[993]
    .float 0.007948, -0.113145, -0.033925, 0.212111, -0.121012, 0.181613, -0.143800, -0.013925, -0.007155, 0.011526  # dense_weights[994]
    .float -0.050008, 0.028855, -0.111645, 0.083766, -0.081340, 0.010801, -0.081975, 0.075698, -0.087692, 0.056508  # dense_weights[995]
    .float 0.054803, -0.129537, 0.133694, 0.096194, -0.140962, -0.009177, 0.029942, -0.159694, 0.062791, -0.107192  # dense_weights[996]
    .float 0.126085, 0.006680, -0.026165, -0.007623, -0.159859, -0.001065, 0.060300, -0.068179, 0.065751, -0.046874  # dense_weights[997]
    .float -0.064310, 0.027691, -0.043446, 0.025269, 0.054320, 0.005404, -0.130517, 0.101122, -0.137157, -0.002845  # dense_weights[998]
    .float 0.110584, -0.051570, 0.070325, 0.084515, -0.022309, 0.077075, 0.020539, -0.019605, 0.017309, -0.052714  # dense_weights[999]
    .float -0.101139, 0.202987, -0.134113, -0.277193, 0.131213, -0.202287, -0.125340, 0.185331, 0.049470, -0.017771  # dense_weights[1000]
    .float 0.018630, -0.093662, 0.038253, 0.111817, -0.106852, 0.058214, 0.000716, -0.059834, 0.059663, 0.030828  # dense_weights[1001]
    .float 0.063832, -0.222752, 0.021430, 0.083311, -0.033221, 0.116811, -0.035248, -0.056592, -0.047691, 0.018595  # dense_weights[1002]
    .float -0.094364, 0.077061, -0.079259, 0.071351, -0.058957, 0.040492, -0.021145, -0.022897, 0.083704, -0.091169  # dense_weights[1003]
    .float 0.066012, -0.087871, 0.082101, 0.041609, -0.011410, 0.006668, 0.101855, -0.084630, 0.035135, -0.171710  # dense_weights[1004]
    .float 0.086987, -0.034866, -0.086940, 0.033803, -0.078965, 0.023439, 0.052803, -0.130347, 0.016751, -0.136622  # dense_weights[1005]
    .float -0.308219, 0.147903, -0.126441, -0.026169, 0.094889, 0.003314, -0.274363, 0.106650, -0.236742, 0.049642  # dense_weights[1006]
    .float 0.096296, 0.028946, -0.084783, 0.049465, -0.052990, 0.082381, 0.023327, 0.036663, -0.004077, -0.002219  # dense_weights[1007]
    .float -0.075840, 0.040414, -0.064826, -0.101516, 0.191744, -0.138541, -0.160942, 0.181695, 0.051984, -0.032734  # dense_weights[1008]
    .float 0.070166, -0.213646, 0.027060, -0.008602, -0.158607, 0.158930, -0.004277, -0.140237, 0.040039, -0.073344  # dense_weights[1009]
    .float 0.065261, -0.288846, 0.103838, 0.153032, -0.133465, 0.133970, -0.045305, -0.058289, 0.011227, 0.029764  # dense_weights[1010]
    .float -0.079921, 0.061264, -0.115924, 0.037013, 0.094720, 0.061012, -0.240005, 0.036756, -0.033745, -0.003643  # dense_weights[1011]
    .float -0.005344, -0.080484, -0.043894, 0.097381, 0.007019, -0.009411, 0.045707, -0.066768, 0.057305, -0.157305  # dense_weights[1012]
    .float -0.015839, 0.095901, -0.022686, 0.036666, 0.002594, -0.026990, 0.122338, -0.032125, 0.074290, -0.159908  # dense_weights[1013]
    .float -0.441176, -0.002214, -0.270033, -0.014049, 0.158245, -0.001116, -0.538363, 0.168852, -0.285252, 0.043146  # dense_weights[1014]
    .float 0.007563, 0.032550, -0.080045, 0.005807, -0.067760, -0.012518, 0.081505, -0.055294, 0.081782, -0.031006  # dense_weights[1015]
    .float -0.128405, -0.038207, -0.010383, -0.088469, 0.121430, -0.105588, 0.010186, 0.082038, -0.064915, 0.002140  # dense_weights[1016]
    .float 0.097061, -0.010722, 0.066458, 0.038055, -0.072286, 0.071319, 0.049676, -0.129696, -0.000552, -0.023149  # dense_weights[1017]
    .float 0.022279, -0.172889, 0.125386, 0.126711, -0.156245, 0.044950, 0.028078, -0.159481, 0.070739, -0.103530  # dense_weights[1018]
    .float -0.139233, 0.053064, -0.038122, 0.127835, 0.050250, 0.093253, -0.301382, -0.006296, 0.074979, 0.005537  # dense_weights[1019]
    .float 0.080834, -0.067796, 0.082176, 0.011186, -0.011315, 0.035640, 0.079791, -0.016942, -0.000125, -0.096444  # dense_weights[1020]
    .float -0.052057, 0.137469, 0.030454, 0.054339, 0.009383, 0.065608, 0.037304, -0.073583, 0.039842, -0.129577  # dense_weights[1021]
    .float -0.297772, 0.101099, -0.250076, 0.136088, 0.165742, -0.011664, -0.396088, 0.095847, -0.229388, 0.016981  # dense_weights[1022]
    .float 0.041174, 0.119771, -0.049955, 0.049607, -0.066810, -0.006835, -0.023351, -0.003151, 0.103253, -0.117777  # dense_weights[1023]
    .float -0.136317, 0.033914, 0.124385, 0.015958, 0.116842, -0.073246, -0.004992, 0.030744, -0.036265, -0.002616  # dense_weights[1024]
    .float -0.018484, 0.020977, 0.129658, -0.040204, -0.032867, -0.005928, 0.045637, -0.098576, -0.040167, -0.042325  # dense_weights[1025]
    .float -0.024353, -0.161640, 0.129035, 0.056079, -0.159525, 0.045205, -0.035875, -0.197529, 0.005767, 0.049922  # dense_weights[1026]
    .float -0.137323, -0.003456, -0.026781, 0.028814, 0.108708, 0.066856, -0.209801, -0.010396, 0.043243, -0.057997  # dense_weights[1027]
    .float -0.119236, 0.082217, 0.039170, 0.071284, -0.057051, -0.091494, 0.018093, -0.021088, 0.099232, -0.003989  # dense_weights[1028]
    .float -0.063512, 0.191752, 0.134872, -0.002460, 0.045497, -0.071685, -0.064263, -0.095150, 0.021104, -0.116901  # dense_weights[1029]
    .float -0.265681, -0.006821, -0.183883, 0.085218, 0.086153, 0.026045, -0.375442, 0.009530, -0.031816, 0.097283  # dense_weights[1030]
    .float 0.001621, 0.120750, -0.015834, -0.046721, -0.000751, 0.070639, 0.009519, -0.130777, 0.046284, 0.047451  # dense_weights[1031]
    .float -0.159856, 0.186376, 0.054516, -0.079813, 0.031529, -0.073120, -0.002388, -0.015078, 0.013356, 0.013711  # dense_weights[1032]
    .float -0.037949, 0.004399, 0.186670, -0.066025, 0.008095, -0.053660, -0.019238, -0.157620, -0.006423, 0.042068  # dense_weights[1033]
    .float -0.032191, -0.086285, 0.264715, -0.017481, -0.044489, -0.054248, -0.059477, -0.210940, -0.009055, -0.019521  # dense_weights[1034]
    .float -0.198073, -0.029020, -0.048361, -0.082458, 0.018716, 0.073221, -0.179674, -0.125355, 0.088147, 0.085250  # dense_weights[1035]
    .float -0.085871, 0.113175, 0.057383, -0.066265, 0.012405, -0.037825, 0.033016, -0.137719, 0.074365, 0.030542  # dense_weights[1036]
    .float -0.095595, 0.081338, 0.032618, -0.135710, 0.099564, -0.002550, 0.034676, -0.229426, 0.059792, 0.017510  # dense_weights[1037]
    .float -0.196521, -0.098647, -0.142698, 0.116102, -0.035685, 0.137430, -0.169755, 0.096476, 0.014444, 0.054829  # dense_weights[1038]
    .float -0.128922, 0.069911, 0.016368, 0.048944, -0.041310, -0.063325, -0.030482, -0.096184, 0.034139, 0.081062  # dense_weights[1039]
    .float -0.120950, 0.052531, -0.071555, 0.038629, 0.011897, -0.040573, 0.062016, -0.084379, -0.019102, -0.017521  # dense_weights[1040]
    .float -0.126230, 0.067229, 0.234574, -0.202773, -0.017548, 0.005946, -0.121039, -0.065323, -0.038069, 0.054246  # dense_weights[1041]
    .float -0.124491, -0.169455, 0.274462, -0.139711, -0.019278, -0.036824, -0.203967, -0.197605, -0.201616, 0.021050  # dense_weights[1042]
    .float -0.167297, 0.073560, 0.012266, 0.035366, -0.042506, 0.232358, -0.208919, -0.000716, -0.127315, 0.060943  # dense_weights[1043]
    .float -0.052453, 0.069525, 0.068606, -0.007266, 0.071072, 0.020092, -0.094478, -0.152979, 0.097245, 0.058059  # dense_weights[1044]
    .float -0.019045, 0.100644, -0.041962, -0.145307, 0.082238, 0.080078, 0.006123, -0.204066, -0.028661, -0.038813  # dense_weights[1045]
    .float -0.226088, 0.008865, 0.046620, 0.051625, -0.164151, 0.165642, -0.145207, -0.013875, -0.063342, 0.070224  # dense_weights[1046]
    .float -0.016160, 0.044289, 0.025350, -0.120768, -0.038027, -0.031902, 0.002686, -0.137541, 0.045004, -0.005045  # dense_weights[1047]
    .float -0.084027, 0.037122, -0.047769, -0.015446, -0.006741, 0.124817, -0.073403, -0.187564, -0.004723, 0.002540  # dense_weights[1048]
    .float -0.042776, -0.025396, 0.189707, -0.025008, -0.035402, -0.005526, -0.083936, -0.096543, -0.162812, 0.059486  # dense_weights[1049]
    .float -0.088452, -0.076954, 0.196878, -0.135398, -0.067635, -0.015992, -0.205827, -0.085468, -0.096926, 0.134423  # dense_weights[1050]
    .float -0.129030, 0.081578, 0.041599, 0.008180, 0.008627, 0.064157, -0.075461, -0.020961, -0.118897, -0.042773  # dense_weights[1051]
    .float -0.138075, -0.062287, 0.019078, -0.097435, -0.054603, 0.012035, -0.088160, -0.224648, 0.025423, 0.086297  # dense_weights[1052]
    .float -0.030326, 0.061590, 0.024563, -0.085310, 0.006935, 0.048402, 0.004620, -0.203550, 0.009045, 0.102964  # dense_weights[1053]
    .float -0.084554, 0.047861, 0.006067, 0.019003, -0.101701, 0.097773, -0.026646, 0.061034, -0.066366, -0.000213  # dense_weights[1054]
    .float -0.105125, 0.089098, -0.028425, -0.014004, 0.019549, 0.029789, -0.029723, -0.106014, 0.050076, 0.013664  # dense_weights[1055]
    .float -0.119966, 0.047722, 0.046261, 0.051701, -0.116871, 0.170617, -0.075934, 0.068999, -0.105368, -0.010315  # dense_weights[1056]
    .float -0.001625, -0.047317, -0.012776, 0.088042, -0.054706, 0.135587, -0.089719, 0.004324, -0.192175, 0.069348  # dense_weights[1057]
    .float -0.027851, -0.009571, 0.051177, 0.040888, -0.102086, 0.118069, -0.080087, 0.021515, -0.100470, 0.085827  # dense_weights[1058]
    .float -0.116602, -0.003849, -0.053536, -0.015115, -0.108471, 0.094939, -0.024110, 0.103161, -0.211292, 0.060926  # dense_weights[1059]
    .float -0.076001, 0.005240, -0.137722, -0.018521, 0.000869, 0.067323, -0.012990, -0.101653, -0.263165, 0.110090  # dense_weights[1060]
    .float -0.115808, -0.010567, -0.064463, 0.084782, -0.045469, -0.023694, -0.071680, 0.088725, -0.170656, 0.076836  # dense_weights[1061]
    .float -0.050138, 0.066893, 0.000006, 0.078164, -0.092624, 0.181408, -0.106397, 0.125169, -0.230308, 0.078248  # dense_weights[1062]
    .float -0.025520, 0.105018, -0.000108, 0.055150, -0.084539, 0.037828, -0.192630, 0.033351, -0.132823, -0.030175  # dense_weights[1063]
    .float -0.166912, 0.044674, 0.002012, -0.016534, -0.028399, 0.244266, -0.067011, -0.017223, -0.059609, -0.011396  # dense_weights[1064]
    .float -0.039537, -0.080922, -0.200724, 0.184326, -0.170060, 0.000703, -0.083380, 0.051827, -0.188750, 0.111001  # dense_weights[1065]
    .float -0.042607, 0.061758, 0.014873, 0.104665, -0.068119, -0.014233, -0.108092, 0.005468, -0.111865, 0.089194  # dense_weights[1066]
    .float -0.050812, -0.101236, -0.030886, 0.023258, -0.076212, -0.030178, -0.145208, 0.096865, -0.109681, 0.126555  # dense_weights[1067]
    .float -0.141745, -0.038397, 0.036034, 0.123965, -0.073908, -0.014610, -0.035268, -0.115086, -0.076135, 0.093373  # dense_weights[1068]
    .float -0.093130, -0.003078, -0.061545, 0.096377, -0.047436, -0.069664, -0.266142, 0.154462, -0.065250, 0.063991  # dense_weights[1069]
    .float -0.020364, -0.037876, -0.042681, 0.013415, -0.009346, -0.092902, -0.191007, 0.155748, -0.106157, 0.115747  # dense_weights[1070]
    .float 0.030788, 0.049570, 0.036996, 0.060133, -0.016178, -0.085392, -0.257318, 0.065537, -0.010382, -0.045013  # dense_weights[1071]
    .float -0.152579, 0.084541, -0.042878, -0.068064, 0.025225, 0.088384, -0.022446, 0.233625, -0.200702, -0.002568  # dense_weights[1072]
    .float -0.227864, -0.102600, -0.211900, 0.241012, -0.109283, 0.007037, -0.157490, 0.157763, -0.178885, 0.188024  # dense_weights[1073]
    .float -0.010017, 0.020522, -0.044835, 0.198181, -0.130885, -0.001981, -0.036965, -0.018791, -0.121051, 0.082499  # dense_weights[1074]
    .float -0.080114, -0.104072, -0.066458, 0.047498, -0.015697, -0.032444, -0.182392, 0.010351, -0.114773, 0.066009  # dense_weights[1075]
    .float -0.059306, 0.087405, -0.014170, 0.199350, -0.079603, -0.169430, -0.164461, 0.096861, -0.171584, -0.069304  # dense_weights[1076]
    .float 0.028901, -0.013864, -0.030025, 0.097035, -0.011751, -0.105582, -0.220611, 0.143223, -0.005508, -0.007772  # dense_weights[1077]
    .float -0.182304, -0.047890, -0.142994, 0.085140, 0.031950, 0.006441, -0.166745, 0.024666, -0.109609, 0.097226  # dense_weights[1078]
    .float 0.084242, 0.040832, 0.008208, 0.110958, -0.039256, -0.075241, -0.116400, 0.137090, -0.044646, 0.008781  # dense_weights[1079]
    .float -0.171749, 0.182262, -0.076832, -0.129322, 0.177282, -0.174171, -0.075757, 0.330248, -0.351997, 0.115281  # dense_weights[1080]
    .float -0.330998, -0.212174, -0.245243, 0.284341, -0.251141, 0.021812, -0.092049, 0.074937, -0.182355, 0.030549  # dense_weights[1081]
    .float 0.049262, -0.078889, -0.122337, 0.180766, -0.205770, 0.097328, -0.093326, -0.026306, -0.105739, -0.028889  # dense_weights[1082]
    .float -0.215624, -0.105977, -0.131195, 0.162274, -0.027953, 0.048354, -0.129703, 0.060441, -0.115180, 0.059212  # dense_weights[1083]
    .float -0.022421, 0.060140, 0.022967, 0.130783, -0.010602, -0.046131, -0.205428, 0.111864, -0.053185, 0.035968  # dense_weights[1084]
    .float 0.080970, 0.014151, -0.043957, 0.084006, -0.046960, -0.045168, -0.212662, -0.035182, -0.043500, 0.000547  # dense_weights[1085]
    .float -0.171464, -0.084689, -0.130217, 0.090932, -0.092771, 0.081271, -0.112825, 0.073206, -0.177200, 0.155408  # dense_weights[1086]
    .float 0.054384, 0.031564, -0.035522, 0.054628, -0.028403, -0.007847, -0.065384, 0.090959, 0.030623, 0.098677  # dense_weights[1087]
    .float -0.152336, 0.099249, -0.328290, -0.207183, 0.173956, -0.185352, -0.137099, 0.328886, -0.248470, 0.159867  # dense_weights[1088]
    .float -0.269617, -0.295325, -0.214812, 0.333288, -0.255288, 0.148581, -0.048008, 0.005971, -0.048613, 0.003040  # dense_weights[1089]
    .float -0.103504, -0.106796, -0.132532, 0.185628, -0.189608, 0.143375, -0.069045, -0.007669, -0.072040, 0.037923  # dense_weights[1090]
    .float -0.089725, -0.117924, -0.140406, 0.140614, -0.058656, 0.050305, -0.032140, 0.045985, -0.000262, 0.103445  # dense_weights[1091]
    .float 0.050369, 0.009744, -0.115311, 0.074018, 0.015154, 0.076245, -0.068290, 0.056533, 0.080382, -0.049478  # dense_weights[1092]
    .float 0.050388, -0.062630, -0.031136, 0.071543, -0.063344, -0.032734, -0.082001, -0.035248, -0.031507, -0.040811  # dense_weights[1093]
    .float -0.090958, -0.111189, -0.130406, 0.006262, -0.045162, 0.039696, -0.064040, 0.090356, -0.171064, 0.158093  # dense_weights[1094]
    .float -0.043123, -0.100338, 0.008881, 0.092780, -0.096801, 0.024203, -0.073583, 0.002163, -0.008668, -0.055933  # dense_weights[1095]
    .float -0.164409, -0.127756, -0.220854, -0.131056, 0.165976, -0.169512, -0.230196, 0.206365, -0.067091, 0.192193  # dense_weights[1096]
    .float -0.197554, -0.336161, -0.065747, 0.251072, -0.264823, 0.187807, -0.143732, -0.037786, 0.039650, -0.037296  # dense_weights[1097]
    .float -0.119328, -0.038170, -0.103268, 0.186115, -0.189412, 0.098859, 0.001624, -0.066001, -0.089046, -0.005569  # dense_weights[1098]
    .float -0.144452, -0.115291, -0.225963, -0.078529, -0.069378, 0.086838, 0.049048, 0.078141, -0.039505, 0.032761  # dense_weights[1099]
    .float 0.021474, -0.133400, -0.079557, 0.039558, 0.013564, -0.009245, -0.145525, 0.063672, 0.088662, 0.023327  # dense_weights[1100]
    .float -0.007250, -0.068573, 0.065886, 0.083263, -0.085446, 0.064764, -0.036835, 0.024458, 0.104370, -0.025262  # dense_weights[1101]
    .float -0.328453, 0.026503, -0.250762, -0.004485, -0.030342, 0.004961, -0.209284, 0.172109, -0.216053, 0.162839  # dense_weights[1102]
    .float -0.000647, 0.004259, -0.018652, 0.118109, -0.076514, 0.131128, -0.076638, 0.045276, -0.002344, 0.021848  # dense_weights[1103]
    .float -0.104121, -0.104245, -0.312473, 0.018589, 0.051712, -0.097897, -0.150799, 0.102297, -0.056821, 0.214377  # dense_weights[1104]
    .float -0.016962, -0.349610, 0.055465, 0.212540, -0.304491, 0.225285, -0.157748, 0.007906, 0.094802, -0.070633  # dense_weights[1105]
    .float -0.169410, 0.052491, -0.043174, 0.112946, -0.144905, 0.129187, -0.164677, 0.077247, -0.118631, -0.010505  # dense_weights[1106]
    .float -0.273028, -0.094750, -0.286090, -0.149025, -0.084056, -0.118456, -0.073406, 0.128250, -0.074309, 0.124843  # dense_weights[1107]
    .float 0.019699, 0.036464, -0.135751, 0.048730, -0.041783, 0.047134, -0.106812, -0.027165, 0.127684, -0.093378  # dense_weights[1108]
    .float -0.006127, 0.002720, -0.011823, 0.008187, -0.032867, 0.011785, 0.014058, -0.027663, 0.098947, -0.046902  # dense_weights[1109]
    .float -0.525850, -0.030922, -0.283200, -0.057836, 0.008965, 0.007810, -0.309149, 0.235627, -0.171546, 0.153996  # dense_weights[1110]
    .float 0.012223, -0.003626, -0.024770, 0.097345, -0.009728, 0.089587, 0.020626, -0.017841, 0.066996, -0.039984  # dense_weights[1111]
    .float -0.044863, -0.141312, -0.183608, -0.028123, -0.020516, 0.062189, -0.159718, 0.111723, 0.116412, 0.147444  # dense_weights[1112]
    .float -0.151177, -0.232306, 0.017955, 0.163184, -0.316349, 0.147950, -0.174996, -0.056344, 0.033862, 0.013493  # dense_weights[1113]
    .float -0.172301, 0.038759, 0.099504, 0.071844, -0.136442, 0.141954, -0.139103, -0.025990, -0.111299, 0.032384  # dense_weights[1114]
    .float -0.376107, 0.120039, -0.217434, -0.081209, -0.072082, -0.108586, -0.176122, 0.173268, -0.033061, 0.067742  # dense_weights[1115]
    .float 0.048385, 0.097308, -0.051366, 0.089826, -0.011787, -0.004484, -0.043924, 0.029399, 0.052221, -0.048773  # dense_weights[1116]
    .float 0.009722, 0.029269, 0.080883, -0.024758, 0.060854, 0.073031, 0.043757, -0.065812, 0.063855, -0.074897  # dense_weights[1117]
    .float -0.405501, 0.070319, -0.206557, 0.038005, 0.033540, 0.017279, -0.311701, 0.244385, -0.218927, 0.085816  # dense_weights[1118]
    .float -0.009294, -0.041544, 0.036665, -0.042902, -0.097364, 0.098759, -0.085939, 0.069864, -0.017753, -0.064395  # dense_weights[1119]
    .float -0.069713, 0.009905, -0.008817, 0.024794, -0.125273, 0.031448, -0.027388, 0.035529, -0.037955, 0.144775  # dense_weights[1120]
    .float -0.235337, -0.122114, 0.002956, -0.004035, -0.254481, 0.098264, -0.231940, -0.022110, -0.061120, 0.122840  # dense_weights[1121]
    .float -0.148185, -0.055200, 0.024661, 0.103328, -0.123923, 0.076784, -0.206952, 0.014989, -0.001412, 0.071349  # dense_weights[1122]
    .float -0.279048, 0.037876, -0.071696, 0.027168, -0.054908, -0.038025, -0.155218, 0.142248, -0.195199, 0.164524  # dense_weights[1123]
    .float -0.063822, 0.127740, 0.056382, -0.070828, -0.021200, -0.029380, -0.008721, -0.054178, 0.075826, -0.030594  # dense_weights[1124]
    .float -0.035219, 0.106918, 0.001737, 0.011274, -0.026381, -0.010687, -0.012827, -0.015805, 0.043990, 0.118841  # dense_weights[1125]
    .float -0.238597, -0.054096, -0.169660, 0.127334, -0.101977, 0.059562, -0.159367, 0.117204, -0.168537, 0.098744  # dense_weights[1126]
    .float -0.143162, -0.010343, 0.108041, -0.013542, -0.013982, 0.016325, -0.085529, -0.023489, 0.034709, 0.051580  # dense_weights[1127]
    .float -0.164183, 0.050390, -0.031945, -0.056999, -0.133395, -0.010541, -0.024127, -0.013589, -0.001563, 0.111459  # dense_weights[1128]
    .float -0.245604, -0.102827, 0.032610, -0.115626, -0.200640, 0.041460, -0.226407, -0.011108, -0.169378, 0.124559  # dense_weights[1129]
    .float -0.105858, -0.030701, 0.049769, -0.077181, -0.157888, 0.117373, -0.252806, 0.103331, -0.149095, 0.051718  # dense_weights[1130]
    .float -0.142868, 0.031031, -0.019243, -0.015515, 0.052687, 0.005125, -0.138426, 0.058792, -0.125656, 0.156493  # dense_weights[1131]
    .float -0.153762, 0.105236, 0.049491, 0.041170, 0.016250, -0.003032, -0.067617, 0.007822, -0.065839, -0.001892  # dense_weights[1132]
    .float -0.072383, 0.088393, 0.008157, -0.046637, 0.011673, -0.077687, -0.011110, -0.134340, -0.008606, 0.028976  # dense_weights[1133]
    .float -0.181907, -0.067251, -0.005057, -0.024488, -0.129237, 0.115124, -0.164296, 0.043339, -0.184841, 0.005724  # dense_weights[1134]
    .float -0.048564, 0.028416, -0.065711, -0.102262, -0.047632, 0.016392, 0.012579, 0.045842, 0.035498, 0.076673  # dense_weights[1135]
    .float -0.113081, 0.050779, -0.026131, -0.067626, -0.075946, 0.080010, -0.014722, -0.057723, 0.001720, 0.087551  # dense_weights[1136]
    .float -0.092096, -0.059463, 0.058067, -0.030556, -0.223738, 0.113991, -0.170016, 0.002987, -0.173488, 0.106468  # dense_weights[1137]
    .float -0.143028, 0.021579, 0.036949, -0.061401, -0.264460, 0.054669, -0.130826, -0.021661, -0.127096, 0.113139  # dense_weights[1138]
    .float -0.188617, -0.030600, 0.008705, -0.043460, -0.006352, 0.118054, -0.130245, 0.012962, -0.203425, -0.009602  # dense_weights[1139]
    .float -0.031278, -0.083537, 0.025166, -0.130262, 0.086818, 0.067345, -0.099630, -0.148216, 0.028817, 0.019209  # dense_weights[1140]
    .float -0.097613, -0.083623, -0.072631, -0.034209, 0.079067, -0.023036, -0.063459, -0.124719, 0.060831, 0.140638  # dense_weights[1141]
    .float -0.068136, 0.017984, 0.087182, 0.014811, -0.096194, 0.128659, -0.124242, -0.001421, -0.112123, -0.062193  # dense_weights[1142]
    .float -0.137076, 0.033090, -0.119721, -0.070825, 0.010323, -0.010976, -0.126413, -0.024516, 0.016327, 0.193559  # dense_weights[1143]
    .float -0.084730, -0.038281, 0.024089, -0.087856, 0.049176, 0.118237, -0.123256, -0.000340, -0.060581, 0.035725  # dense_weights[1144]
    .float -0.107484, 0.027988, 0.044517, -0.070098, -0.077982, 0.147568, -0.141360, 0.062938, -0.137230, 0.077907  # dense_weights[1145]
    .float -0.091326, -0.050863, 0.063277, -0.008584, -0.135710, 0.106675, -0.093072, 0.011470, -0.080291, -0.013720  # dense_weights[1146]
    .float -0.127165, 0.080283, 0.014378, -0.002842, -0.137824, 0.073751, -0.009840, 0.051795, -0.120687, 0.040497  # dense_weights[1147]
    .float -0.095261, -0.072066, 0.037034, -0.035275, 0.068328, 0.059915, -0.127580, -0.110199, -0.021162, 0.115332  # dense_weights[1148]
    .float -0.102646, 0.042649, -0.054880, 0.011204, -0.030020, 0.044189, -0.191696, -0.126215, -0.067745, 0.129520  # dense_weights[1149]
    .float -0.138149, 0.071385, 0.092821, -0.046511, -0.088703, 0.124388, 0.000842, -0.010875, -0.150524, -0.070720  # dense_weights[1150]
    .float -0.083616, -0.023687, 0.004885, -0.052437, -0.018558, 0.025708, -0.208173, -0.053941, -0.003236, 0.122852  # dense_weights[1151]

# DENSE_WEIGHTS END norm_loop

# 10 Dense biases input_matrix

.global bias

## DENSE_BIAS BEGIN
dense_bias:
    .float -0.041667, 0.033751, 0.030045, 0.006700, -0.050259, 0.085334, -0.033102, 0.026402, -0.079888, 0.002525  # dense_bias[0:10]
## DENSE_BIAS END

.global weights_size
Dense_weights_size: .word 11520
.global biases_size
Dense_biases_size: .word 10


# ## Convolutional Layer Filters
.global conv_filters

## FILTER BEGIN
# 5x5x1x8 Convolution Filter Weights
conv_filters:
# Filter 1 Weights:
    .float 0.300339, 0.227408, -0.236021, -0.446589, -0.396676
    .float 0.401512, 0.174840, -0.068394, -0.537320, -0.266781
    .float 0.344698, 0.308167, 0.042030, -0.603082, -0.376260
    .float 0.400408, 0.479303, 0.147943, -0.377725, -0.361802
    .float 0.131693, 0.429212, 0.314070, 0.053970, -0.165925
# Filter 2 Weights:
    .float -0.248271, -0.452428, -0.380840, -0.325791, -0.150723
    .float 0.140508, -0.198228, -0.150268, -0.077117, -0.039421
    .float 0.127301, 0.338143, 0.131038, 0.197601, 0.030741
    .float 0.310692, 0.271372, 0.265915, 0.208181, 0.167045
    .float 0.058608, -0.046019, 0.176440, 0.069233, 0.140819
# Filter 3 Weights:
    .float -0.290180, -0.464000, -0.378948, -0.119563, -0.204608
    .float -0.506382, -0.345332, -0.378812, -0.134977, -0.211963
    .float -0.015084, 0.153549, -0.040216, -0.148484, -0.152374
    .float 0.482564, 0.331063, 0.253583, 0.265747, 0.301826
    .float 0.211166, 0.281555, 0.228046, 0.297610, 0.016170
# Filter 4 Weights:
    .float -0.149720, 0.026225, -0.023406, 0.158984, 0.155360
    .float -0.112656, -0.262022, -0.086311, 0.014723, 0.215976
    .float -0.391579, -0.311004, 0.037853, 0.359984, 0.308057
    .float -0.467311, -0.049964, 0.281449, 0.355260, 0.053169
    .float 0.061342, 0.379726, 0.333479, 0.053468, -0.199379
# Filter 5 Weights:
    .float 0.019169, 0.267373, 0.069999, -0.142359, -0.304575
    .float 0.086512, 0.308514, 0.079640, -0.218450, -0.588875
    .float 0.001121, -0.040538, -0.044429, -0.159208, -0.197143
    .float -0.019575, -0.153058, 0.243038, 0.390175, 0.339090
    .float 0.135671, -0.082696, 0.098810, 0.432103, 0.124045
# Filter 6 Weights:
    .float -0.009802, 0.125539, 0.223976, 0.025631, 0.026455
    .float 0.105429, 0.230694, 0.323925, 0.316107, 0.145338
    .float 0.062867, 0.222572, 0.066369, 0.033060, 0.046972
    .float 0.059091, -0.284220, -0.014942, 0.086095, 0.204017
    .float -0.722000, -0.537082, -0.376672, -0.030316, -0.086823
# Filter 7 Weights:
    .float -0.389260, -0.460942, -0.696670, 0.088243, 0.236628
    .float -0.574489, -0.605922, -0.197509, 0.192607, 0.465190
    .float -0.632420, -0.374857, 0.159006, 0.381875, 0.351684
    .float -0.628080, -0.273170, 0.266438, 0.410701, 0.178492
    .float -0.653007, -0.030996, 0.418446, 0.247709, 0.187084
# Filter 8 Weights:
    .float 0.113788, 0.076092, 0.235073, 0.074814, 0.076887
    .float 0.029188, -0.086627, 0.181658, 0.056439, -0.146885
    .float 0.098740, 0.141947, 0.158142, -0.037740, 0.159212
    .float 0.105803, 0.099431, 0.050659, -0.066341, -0.088222
    .float 0.052213, 0.063916, 0.257254, 0.028423, 0.068510

# FILTER END

.global conv_biases

## FILTER_BIAS BEGIN
filter_bias:
    .float 0.046092, 0.116175, 0.081196, -0.008723, -0.001550, 0.000569, 0.138644, -0.060249  # filter_bias[0:8] 
## FILTER_BIAS END

.global conv_filters_size
conv_filters_size: .word 200
.global conv_biases_size
conv_biases_size: .word 8

## INPUTS BEGIN conv_output
.section .data
.global input_image

input_matrix:
# Image is Scaled between 0 and 255

## INPUT_MATRIX BEGIN
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[0]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[1]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[2]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[3]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[4]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.011765, 0.070588, 0.070588, 0.070588, 0.494118, 0.533333, 0.686275, 0.101961, 0.650980, 1.000000, 0.968627, 0.498039, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[5]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.117647, 0.141176, 0.368627, 0.603922, 0.666667, 0.992157, 0.992157, 0.992157, 0.992157, 0.992157, 0.882353, 0.674510, 0.992157, 0.949020, 0.764706, 0.250980, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[6]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.192157, 0.933333, 0.992157, 0.992157, 0.992157, 0.992157, 0.992157, 0.992157, 0.992157, 0.992157, 0.984314, 0.364706, 0.321569, 0.321569, 0.219608, 0.152941, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[7]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.070588, 0.858824, 0.992157, 0.992157, 0.992157, 0.992157, 0.992157, 0.776471, 0.713725, 0.968627, 0.945098, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[8]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.313725, 0.611765, 0.419608, 0.992157, 0.992157, 0.803922, 0.043137, 0.000000, 0.168627, 0.603922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[9]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.054902, 0.003922, 0.603922, 0.992157, 0.352941, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[10]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.545098, 0.992157, 0.745098, 0.007843, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[11]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.043137, 0.745098, 0.992157, 0.274510, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[12]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.137255, 0.945098, 0.882353, 0.627451, 0.423529, 0.003922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[13]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.317647, 0.941176, 0.992157, 0.992157, 0.466667, 0.098039, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[14]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.176471, 0.729412, 0.992157, 0.992157, 0.588235, 0.105882, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[15]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.062745, 0.364706, 0.988235, 0.992157, 0.733333, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[16]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.976471, 0.992157, 0.976471, 0.250980, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[17]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.180392, 0.509804, 0.717647, 0.992157, 0.992157, 0.811765, 0.007843, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[18]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.152941, 0.580392, 0.898039, 0.992157, 0.992157, 0.992157, 0.980392, 0.713725, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[19]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.094118, 0.447059, 0.866667, 0.992157, 0.992157, 0.992157, 0.992157, 0.788235, 0.305882, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[20]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.090196, 0.258824, 0.835294, 0.992157, 0.992157, 0.992157, 0.992157, 0.776471, 0.317647, 0.007843, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[21]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.070588, 0.670588, 0.858824, 0.992157, 0.992157, 0.992157, 0.992157, 0.764706, 0.313725, 0.035294, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[22]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.215686, 0.674510, 0.886275, 0.992157, 0.992157, 0.992157, 0.992157, 0.956863, 0.521569, 0.043137, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[23]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.533333, 0.992157, 0.992157, 0.992157, 0.831373, 0.529412, 0.517647, 0.062745, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[24]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[25]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[26]
  .float 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 # input_matrix[27]
## INPUT_MATRIX END

