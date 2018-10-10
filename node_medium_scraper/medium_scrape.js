const { getPosts, getUser } = require('medium-posts')
var request = require('request');
var fs = require('fs');

function getMediumPosts(userName, numOfArticles) {
  getPosts(userName, numOfArticles).then((posts) => {
      let fileCount = 0;

      if (posts.length > 0) {
        for (post of posts) {
          fileCount += 1;

          let logger = fs.createWriteStream('./data/' + userName + fileCount + '.txt', {
            flags: 'w+'
          });

          if (post.url) {
            request(post.url + '?format=json', function(error, response, body) {

              let b = body;
              b = b.replace("])}while(1);</x>", "");

              bodyJson = JSON.parse(b);

              // Account for 2 versions of articile that I have come accross
              if (bodyJson.payload.value) {
                paras = bodyJson.payload.value.content.bodyModel.paragraphs;
              } else {
                paras = bodyJson.payload.post.content.bodyModel.paragraphs;
              }

              for (paragraph of paras) {
                logger.write(`${paragraph.text}\n`);
              }

            });
          }
        }
      }
    })
    .catch((err) => {
      console.log('Error getMediumPosts ', err);
    });
}

/*
 * Params: userName from Medium and the number
 * of desired articles.  It will create a new
 * file for each article in data/
 */
getMediumPosts('george.seif94', 3);