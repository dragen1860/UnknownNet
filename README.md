# UnknownNet

| Configuration                             	| time for 0, accuracy 	| time for 1  	| time for 2  	|    time for 3 |  time for 4   |
|-------------------------------------------	|----------------------	|-------------	|-------------	|-------------	|-----------	|
| loss backward recursively, wrong version  	| 122.4s, 90.4%        	|             	|             	|             	|           	|
| loss backward recursively, reversed order 	| 106s, 92%            	| 155s, 92.4% 	| 269s, 91.4% 	| 378s, 91.8% 	| 688s, 93% 	|
| with last loss backward only              	| 157s, 90.08%         	| 227s, 93.8% 	| 397s, 92.2% 	|             	|           	|
| loss backward reursively, normal order    	| 184s, 91.4%          	| 220s, 90.2% 	| 442s, 91.4% 	| 495s, 90.4% 	|           	|
| loss backward recurisvely, revsed, one step  	|  85s, 90.6%        	| 135.9s, 90.8% | 281s, 94.6%	|  243s, 91.6%  | 455s, 90.6%   |
|                                           	|                      	|             	|             	|             	|           	|
|                                           	|                      	|             	|             	|             	|           	|
|                                           	|                      	|             	|             	|             	|           	|
