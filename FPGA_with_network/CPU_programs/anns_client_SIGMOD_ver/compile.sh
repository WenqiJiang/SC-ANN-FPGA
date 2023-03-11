rm network_send
gcc network_send.c -lpthread -o network_send
rm network_recv
gcc network_recv.c -lpthread -o network_recv
rm network_send_recv
gcc network_send_recv.c -lpthread -o network_send_recv
rm network_send_recv_client
# gcc network_send_recv_client.c -lpthread -o network_send_recv_client
g++ network_send_recv_client.cpp -lpthread -o network_send_recv_client
rm network_recv_send_server
# gcc network_recv_send_server.c -lpthread -o network_recv_send_server
g++ network_recv_send_server.cpp -lpthread -o network_recv_send_server
rm network_send_recv_client_sync
g++ network_send_recv_client_sync.cpp -lpthread -o network_send_recv_client_sync
rm network_send_recv_client_sync_aux_thread
g++ network_send_recv_client_sync_aux_thread.cpp -lpthread -o network_send_recv_client_sync_aux_thread
rm network_send_recv_client_sync_more_start
g++ network_send_recv_client_sync_more_start.cpp -lpthread -o network_send_recv_client_sync_more_start
rm anns_client
g++ anns_client.cpp -lpthread -o anns_client
rm anns_single_client_async
g++ anns_single_client_async.cpp -lpthread -o anns_single_client_async
rm anns_client_1_to_3_connections
g++ anns_client_1_to_3_connections.cpp -lpthread -o anns_client_1_to_3_connections
