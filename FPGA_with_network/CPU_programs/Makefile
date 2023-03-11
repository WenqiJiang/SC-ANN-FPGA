# g++ -Wall anns_client.cpp -lpthread  -o anns_client
CC = g++
CLAGS=-Wall
LINK = -lpthread

all: anns_client anns_client_tolerant anns_client_non_blocking

anns_client: anns_client.cpp
	${CC} ${CLAGS} anns_client.cpp ${LINK} -o anns_client

anns_client_tolerant: anns_client_tolerant.cpp
	${CC} ${CLAGS} anns_client_tolerant.cpp ${LINK} -o anns_client_tolerant

anns_client_non_blocking: anns_client_non_blocking.cpp
	${CC} ${CLAGS} anns_client_non_blocking.cpp ${LINK} -o anns_client_non_blocking

.PHONY: clean, cleanall

cleanall: clean

clean:
	rm anns_client anns_client_tolerant anns_client_tolerant
