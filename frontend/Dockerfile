# Stage 1: Build the React App
FROM node:18-alpine as build
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve with Nginx (Production Grade)
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
# Expose port 80 (standard web port)
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]